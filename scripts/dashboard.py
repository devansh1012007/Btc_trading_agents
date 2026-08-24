#!/usr/bin/env python
"""
TUI Dashboard (Phase 5, Module G / §10 — terminal variant, M5.10).

Six views answering the plan's questions, all reading the SAME artifacts
diagnose.py reads (decision-trace JSONL, faults.jsonl, registry.db,
backtest reports) so CLI numbers and dashboard numbers can never drift:

  [1] Signal Path    — per-stage pass rates: how far bars get through the pipeline
  [2] Decision Funnel — where bars actually STOP (terminal-stage breakdown)
  [3] Trades          — journal tail, today's PnL/win-rate, open positions
  [4] Performance     — latest backtest report vs Project_Plan.md criteria + baselines
  [5] Calibration     — confidence reliability buckets per expert + router
  [6] Risk & Registry — circuit breakers, risk state, faults, model versions, experiments

Keys: 1..6 switch views · r refresh · q quit. Auto-refreshes every 5s.
Headless smoke mode: python scripts/dashboard.py --once

Requires: textual (pip install textual)
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.diagnostics import dashboard_data as dd  # noqa: E402


# ---------------------------------------------------------------------------
# Shared Rich renderables (used by live TUI and --once mode alike)
# ---------------------------------------------------------------------------

def _fmt(v, suffix="", nd=2):
    if v is None:
        return "—"
    return f"{v:.{nd}f}{suffix}"


def _bar(pct: float, width: int = 24) -> str:
    filled = max(0, min(width, int(round(pct / 100 * width))))
    return "█" * filled + "·" * (width - filled)


def build_status_line(snap: dict) -> str:
    st = snap["status"]
    db = snap["db"]
    kill = "[bold red]KILL SWITCH ACTIVE[/]" if st["kill_switch_active"] else "[green]kill-switch clear[/]"
    if not db["db_present"]:
        dbtxt = "[yellow]registry.db absent[/]"
    else:
        dbtxt = f"breakers active: [red]{len(db['active_breakers'])}[/]" if db["active_breakers"] \
            else "[green]no breakers active[/]"
        dbtxt += f" · open positions: {db['open_trades']}"
    return f"{kill} · {dbtxt} · traces loaded: {snap['n_traces']}"


def build_signal_path_view(snap: dict):
    from rich.console import Group
    from rich.table import Table
    from rich import box

    rates = Table(title="Stage pass rates (bars reaching each stage)", box=box.SIMPLE_HEAVY)
    rates.add_column("stage", style="cyan")
    rates.add_column("reached", justify="right")
    rates.add_column("passed", justify="right")
    rates.add_column("pass %", justify="right")
    rates.add_column("", overflow="crop")
    for r in snap["stage_rates"]:
        color = "green" if r["pass_pct"] >= 60 else ("yellow" if r["pass_pct"] >= 25 else "red")
        rates.add_row(r["stage_name"], str(r["reached"]), str(r["passed"]),
                      f"[{color}]{r['pass_pct']:.1f}%[/]", _bar(r["pass_pct"]))

    funnel = Table(title="Terminal funnel — where bars stop", box=box.SIMPLE_HEAVY)
    funnel.add_column("terminal stage")
    funnel.add_column("count", justify="right")
    funnel.add_column("share", justify="right")
    funnel.add_column("", overflow="crop")
    for r in snap["funnel"]:
        style = "green" if r["stage"] == "executed" else ("red" if r["stage"] == "circuit_breaker" else "white")
        funnel.add_row(f"[{style}]{r['label']}[/]", str(r["count"]),
                       f"{r['pct']:.1f}%", _bar(r["pct"], 20))

    if not snap["stage_rates"] and not snap["funnel"]:
        hint = ("\n[dim]No decision traces yet.[/] Run a replay first:\n"
                "  python scripts/run.py --mode historical --input data/raw/BTCUSDT.csv --env paper\n"
                "  python scripts/full_stack_walk_forward_backtest.py --synthetic 900 --relax-liquidity")
        from rich.text import Text
        return Text(hint)
    return Group(rates, funnel)


def build_trades_view(snap: dict):
    from rich.console import Group
    from rich.table import Table
    from rich import box
    from datetime import datetime, timezone

    t = Table(title=f"Trade journal tail (open positions: {snap['db']['open_trades']})",
              box=box.SIMPLE_HEAVY)
    for col, just in [("entry", "left"), ("exit", "left"), ("expert", "cyan"),
                      ("regime", "magenta"), ("dir", "center"), ("entry px", "right"),
                      ("exit px", "right"), ("reason", "yellow"), ("pnl %", "right"),
                      ("pnl $", "right")]:
        t.add_column(col, justify=just)
    for tr in snap["db"]["recent_trades"]:
        fmt_ts = lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%m-%d %H:%M") if ts else "—"
        pnl = tr.get("realized_pnl_pct")
        pnl_s = f"[{'green' if pnl and pnl > 0 else 'red'}]{_fmt(pnl, '', 3)}[/]" if pnl is not None else "open"
        usd = tr.get("realized_pnl_usdt")
        t.add_row(fmt_ts(tr.get("entry_timestamp")), fmt_ts(tr.get("exit_timestamp")),
                  tr.get("expert_used") or "—", tr.get("regime_at_entry") or "—",
                  "+" if tr.get("direction") == 1 else "-",
                  _fmt(tr.get("entry_price")), _fmt(tr.get("exit_price")),
                  tr.get("exit_reason") or "[dim]open[/]", pnl_s, _fmt(usd))
    if not snap["db"]["recent_trades"]:
        from rich.text import Text
        return Text("[dim]trade_journal empty or registry.db absent — no trades recorded yet.[/]")
    return Group(t)


def build_performance_view(snap: dict):
    from rich.console import Group
    from rich.table import Table
    from rich import box

    rep = snap["report"]
    if not rep.get("present"):
        from rich.text import Text
        return Text("[dim]No backtest report found in data/reports/.[/]\n"
                    "Generate one:\n"
                    "  python scripts/full_stack_walk_forward_backtest.py "
                    "--input data/raw/BTCUSDT.csv --env paper\n"
                    "  python scripts/full_stack_walk_forward_backtest.py --synthetic 900 --relax-liquidity")

    agg = rep["aggregate"]
    head = Table(title=f"Aggregate — report {rep['config_hash']} ({rep['generated_at'][:19]}Z)",
                 box=box.SIMPLE_HEAVY, show_header=False)
    head.add_column("k"); head.add_column("v")
    rows = [
        ("total return", _fmt(agg.get("total_return_pct"), "%")),
        ("trades", str(agg.get("n_trades", 0))),
        ("win rate", _fmt(agg.get("win_rate_pct"), "%", 1)),
        ("expectancy/trade", _fmt(agg.get("expectancy_per_trade_pct"), "%", 4)),
        ("profit factor", _fmt(agg.get("profit_factor"))),
        ("sharpe / sortino (per-trade)", f"{_fmt(agg.get('sharpe_ratio'))} / {_fmt(agg.get('sortino_ratio'))}"),
        ("max drawdown", _fmt(agg.get("max_drawdown_pct"), "%")),
    ]
    for k, v in rows:
        head.add_row(f"[dim]{k}[/]", f"[bold]{v}[/]")

    crit = Table(title="Project_Plan.md success criteria", box=box.SIMPLE_HEAVY)
    crit.add_column("verdict", justify="center")
    crit.add_column("criterion")
    crit.add_column("target", overflow="fold")
    crit.add_column("actual", justify="right")
    for c in rep["criteria"]:
        mark = "[green]✓ PASS[/]" if c["pass"] else "[red]✗ FAIL[/]"
        actual = c.get("actual")
        base = c.get("baseline_actual")
        s = _fmt(actual, "%", 3) if isinstance(actual, float) else str(actual)
        if base is not None:
            s += f" [dim](mom: {_fmt(base, '%', 3) if isinstance(base, float) else base})[/]"
        crit.add_row(mark, c["criterion"], c["target"], s)

    bl = Table(title="Baselines", box=box.SIMPLE_HEAVY)
    for col, j in [("strategy", "left"), ("n", "right"), ("ret %", "right"),
                   ("win %", "right"), ("exp %", "right"), ("sharpe", "right"), ("dd %", "right")]:
        bl.add_column(col, justify=j)
    name_map = {"buy_and_hold": "buy & hold", "simple_momentum": "simple momentum",
                "naive_router": "naive router", "unfiltered": "no opp-filter"}
    for b in rep["baselines"]:
        nm = name_map.get(b["name"], b["name"])
        bl.add_row(nm, str(b["n_trades"] or 0), _fmt(b["total_return_pct"]),
                   _fmt(b["win_rate_pct"], "", 1), _fmt(b["expectancy_per_trade_pct"], "", 3),
                   _fmt(b["sharpe_ratio"]), _fmt(b["max_drawdown_pct"]))
    verd = "\n".join(f"[dim]{k.replace('_', ' ')}:[/] [bold]{v}[/]"
                     for k, v in rep.get("verdicts", {}).items())

    parts = [head, crit, bl]
    if verd:
        from rich.text import Text
        parts.append(Text(verd))
    wins = [w["return_pct"] for w in rep["windows"] if w["return_pct"] is not None]
    if len(wins) >= 2:
        parts.append(("window_returns", wins))  # caller turns into Sparkline
    from rich.console import Group as G
    return G(*parts[:-1]) if parts and isinstance(parts[-1], tuple) else G(*parts)


def window_returns(snap: dict):
    rep = snap["report"]
    if not rep.get("present"):
        return []
    return [w["return_pct"] for w in rep["windows"] if w["return_pct"] is not None]


def build_calibration_view(snap: dict):
    from rich.table import Table
    from rich.console import Group
    from rich import box

    rep = snap["report"]
    if not rep.get("present"):
        from rich.text import Text
        return Text("[dim]No backtest report — calibration needs a run's reliability data.[/]")

    parts = []
    cal = rep.get("calibration_experts", {})
    for expert, info in sorted(cal.items()):
        t = Table(title=f"{expert}  (weighted |pred−realized| = {_fmt(info.get('score'), '', 3)})",
                  box=box.SIMPLE, show_header=True)
        for col, j in [("bucket", "left"), ("n", "right"), ("avg pred", "right"),
                       ("realized win %", "right"), ("gap", "right")]:
            t.add_column(col, justify=j)
        for b in info["buckets"]:
            if b["n"] == 0:
                continue
            gap = abs(b["avg_predicted_confidence"] - b["realized_win_rate_pct"] / 100) \
                if b["avg_predicted_confidence"] is not None else None
            gcol = "green" if (gap is not None and gap < 0.15) else ("yellow" if gap < 0.3 else "red")
            t.add_row(b["bucket"], str(b["n"]), _fmt(b["avg_predicted_confidence"], "", 3),
                      _fmt(b["realized_win_rate_pct"], "", 1),
                      f"[{gcol}]{_fmt(gap, '', 3)}[/]")
        parts.append(t)

    rc = rep.get("router_calibration", {})
    rt = Table(title=f"router — executed trades (score = {_fmt(rc.get('score'), '', 3)})",
               box=box.SIMPLE)
    for col, j in [("bucket", "left"), ("n", "right"), ("avg pred", "right"), ("win %", "right")]:
        rt.add_column(col, justify=j)
    any_rows = False
    for b in rc.get("buckets", []):
        if b["n"]:
            any_rows = True
            rt.add_row(b["bucket"], str(b["n"]), _fmt(b["avg_predicted_confidence"], "", 3),
                       _fmt(b["realized_win_rate_pct"], "", 1))
    if any_rows:
        parts.append(rt)

    if not parts:
        from rich.text import Text
        return Text("[dim]Report present but contained no calibration buckets.[/]")
    return Group(*parts)


def build_risk_view(snap: dict):
    from rich.table import Table
    from rich.console import Group
    from rich.text import Text
    from rich import box

    db = snap["db"]
    if not db["db_present"]:
        return Text("[dim]registry.db absent — nothing to show yet.[/]")

    parts = []
    ab = Table(title="Active breakers", box=box.SIMPLE_HEAVY)
    ab.add_column("breaker", style="bold red")
    ab.add_column("tripped_at"); ab.add_column("reason", overflow="fold")
    for b in db["active_breakers"]:
        ab.add_row(b["breaker_type"], str(b.get("tripped_at")), str(b.get("reason")))
    if not db["active_breakers"]:
        ab.add_row("[green]none[/]", "", "")

    hist = Table(title="Breaker history (latest 20)", box=box.SIMPLE)
    for col, j in [("breaker", "left"), ("active", "center"), ("tripped_at", "left"),
                   ("reset_at", "left")]:
        hist.add_column(col, justify=j)
    for b in db["breaker_history"]:
        act = "[red]ACTIVE[/]" if b.get("active") else "[dim]reset[/]"
        hist.add_row(b["breaker_type"], act, str(b.get("tripped_at")), str(b.get("reset_at")))

    rs = Table(title="risk_state", box=box.SIMPLE, show_header=False)
    rs.add_column("key", style="cyan"); rs.add_column("value")
    for k in sorted(db["risk_state"]):
        rs.add_row(k, str(db["risk_state"][k]))

    parts += [ab, hist, rs]

    fl = Table(title=f"Faults (tail of {len(db.get('_faults_all', [])) or 'data/logs/faults.jsonl'})",
               box=box.SIMPLE)
    fl.add_column("time"); fl.add_column("type", style="red"); fl.add_column("message", overflow="fold")
    faults = dd.load_faults()
    for f in faults[-8:]:
        fl.add_row(time.strftime("%m-%d %H:%M:%S", time.gmtime(f.get("timestamp", 0))),
                   f.get("exception_type", "?"),
                   (f.get("exception_message", "") or "")[:90])
    if faults:
        parts.append(fl)
    return Group(*parts)


def build_registry_view(snap: dict):
    from rich.table import Table
    from rich.console import Group
    from rich.text import Text
    from rich import box

    db = snap["db"]
    if not db["db_present"]:
        return Text("[dim]registry.db absent.[/]")
    parts = []

    mv = Table(title="Model versions (latest 30)", box=box.SIMPLE_HEAVY)
    for col, j in [("component", "cyan"), ("version_id", "left"), ("status", "left"),
                   ("granularity", "center"), ("trained_at", "left")]:
        mv.add_column(col, justify=j)
    for m in db["models"]:
        st = m["status"]
        scolor = {"approved_active": "green", "pending_approval": "yellow",
                  "superseded": "dim", "rejected": "red"}.get(st, "white")
        mv.add_row(m["component"], m["version_id"], f"[{scolor}]{st}[/]",
                   m.get("data_granularity") or "—", str(m.get("trained_at", ""))[:19])
    if db["models"]:
        parts.append(mv)

    ex = Table(title="Experiments (latest 15)", box=box.SIMPLE_HEAVY)
    ex.add_column("experiment_id"); ex.add_column("type", justify="center")
    ex.add_column("component", style="cyan"); ex.add_column("cfg", justify="center")
    ex.add_column("started_at"); ex.add_column("metrics", overflow="fold")
    for e in db["experiments"]:
        mjson = e.get("metrics_json")
        extra = ""
        if mjson:
            try:
                mm = json.loads(mjson)
                if "success_criteria_passed" in mm:
                    extra = f"criteria {mm['success_criteria_passed']}/4"
                elif "aggregate" in mm and mm["aggregate"].get("n_trades") is not None:
                    extra = f"trades {mm['aggregate'].get('n_trades')}"
            except json.JSONDecodeError:
                extra = "?"
        ex.add_row(e["experiment_id"], e["run_type"], e.get("component") or "—",
                   e.get("config_hash") or "—", str(e.get("started_at", ""))[:19], extra)
    if db["experiments"]:
        parts.append(ex)

    if not parts:
        return Text("[dim]model_versions and experiments are both empty — train/backtest something first.[/]")
    return Group(*parts)


VIEW_BUILDERS = [
    ("Signal Path", build_signal_path_view),
    ("Funnel", lambda snap: build_signal_path_view(snap)),   # combined pane; kept for tab title parity
    ("Trades", build_trades_view),
    ("Performance", build_performance_view),
    ("Calibration", build_calibration_view),
    ("Risk & Registry", build_registry_view),
]


def build_all_views(snap: dict):
    """One rendering path for --once and the live app."""
    return {
        "Signal Path": build_signal_path_view(snap),
        "Trades": build_trades_view(snap),
        "Performance": build_performance_view(snap),
        "Calibration": build_calibration_view(snap),
        "Risk & Registry": build_risk_view(snap),
        "Registry": build_registry_view(snap),
    }


# ---------------------------------------------------------------------------
# Live TUI
# ---------------------------------------------------------------------------

def run_tui(refresh_seconds: float = 5.0):
    try:
        from textual.app import App, ComposeResult
        from textual.widgets import Header, Footer, Static, TabbedContent, TabPane, Sparkline
        from textual.containers import Vertical
    except ImportError:
        raise SystemExit(
            "textual is required for the live dashboard:\n"
            "  pip install textual\n"
            "(or use the headless renderer: python scripts/dashboard.py --once)")

    class DashboardApp(App):
        TITLE = "BTC Trading Agents — Signal Console"
        SUB_TITLE = "decision traces · journal · backtests · registry"
        BINDINGS = [
            ("q", "quit", "Quit"),
            ("r", "refresh", "Refresh now"),
            ("1", "tab(0)", "Path"),
            ("2", "tab(1)", "Trades"),
            ("3", "tab(2)", "Perf"),
            ("4", "tab(3)", "Calib"),
            ("5", "tab(4)", "Risk"),
            ("6", "tab(5)", "Registry"),
        ]

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Static(id="statusline", classes="statusline")
            with TabbedContent(initial="path"):
                with TabPane("① Signal Path", id="path"):
                    yield Static(id="view-path", expand=True)
                with TabPane("② Trades", id="trades"):
                    yield Static(id="view-trades", expand=True)
                with TabPane("③ Performance", id="perf"):
                    with Vertical():
                        yield Sparkline(id="window-spark", summary_function=max)
                        yield Static(id="view-perf", expand=True)
                with TabPane("④ Calibration", id="calib"):
                    yield Static(id="view-calib", expand=True)
                with TabPane("⑤ Risk & Faults", id="risk"):
                    yield Static(id="view-risk", expand=True)
                with TabPane("⑥ Model Registry", id="registry"):
                    yield Static(id="view-registry", expand=True)
            yield Footer()

        def on_mount(self):
            self.set_interval(refresh_seconds, self.action_refresh)
            self.action_refresh()

        def action_refresh(self) -> None:
            snap = dd.build_snapshot()
            try:
                self.query_one("#statusline", Static).update(build_status_line(snap))
                mapping = {
                    "#view-path": build_signal_path_view(snap),
                    "#view-trades": build_trades_view(snap),
                    "#view-calib": build_calibration_view(snap),
                    "#view-risk": build_risk_view(snap),
                    "#view-registry": build_registry_view(snap),
                }
                for sel, renderable in mapping.items():
                    self.query_one(sel, Static).update(renderable)
                perf_parts = build_performance_view(snap)
                self.query_one("#view-perf", Static).update(perf_parts)
                wins = window_returns(snap)
                if wins:
                    try:
                        self.query_one("#window-spark", Sparkline).data = wins
                    except Exception:
                        pass
            except Exception as e:  # never let one bad artifact kill the TUI
                self.query_one("#statusline", Static).update(
                    f"[red]render error:[/] {type(e).__name__}: {e}")

        def action_tab(self, index: int) -> None:
            self.query_one(TabbedContent).active = \
                ["path", "trades", "perf", "calib", "risk", "registry"][index]

    CSS = """
    Screen { background: #0d1117; }
    .statusline { dock: top; height: 1; padding: 0 1; background: #161b22;
                  border-bottom: solid #30363d; }
    TabbedContent { height: 1fr; }
    Static { padding: 0 1; }
    """
    DashboardApp.CSS = CSS
    DashboardApp.run()


def run_once() -> None:
    from rich.console import Console
    console = Console()
    snap = dd.build_snapshot()
    console.rule("[bold]BTC Trading Agents — Signal Console (--once)[/]")
    console.print(build_status_line(snap))
    views = build_all_views(snap)
    titles = {
        "Signal Path": "SIGNAL PATH & DECISION FUNNEL",
        "Trades": "TRADE JOURNAL",
        "Performance": "PERFORMANCE (latest backtest report)",
        "Calibration": "CONFIDENCE CALIBRATION",
        "Risk & Registry": "RISK STATE, BREAKERS & FAULTS",
        "Registry": "MODEL REGISTRY & EXPERIMENTS",
    }
    wins = window_returns(snap)
    if len(wins) >= 2:
        console.print("\n[bold]Window returns[/]: " + " ".join(f"{w:+.2f}" for w in wins))
    for title, renderable in views.items():
        console.rule(f"[bold cyan]{titles[title]}[/]")
        console.print(renderable)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--once", action="store_true",
                        help="Render one static frame to stdout and exit (TTY-less)")
    parser.add_argument("--refresh-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.once:
        run_once()
    else:
        run_tui(args.refresh_seconds)


if __name__ == "__main__":
    main()
