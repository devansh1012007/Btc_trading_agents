"""
The minimum-trade-duration rule, as ONE object.

``HoldPolicy`` turns ``trade_duration.min_trade_seconds`` (default **10 s**)
into the integers the rest of the system needs, and it is the only place
that arithmetic lives:

* the triple-barrier LABELER (training) uses ``min_hold_bars``;
* ``check_position_exit`` (the harness, the live loop and the GUI replay,
  all through ``process_bar``) uses ``min_hold_bars`` AND the wall-clock
  ``min_trade_seconds``;
* the experts and every other ML component compare a model's recorded
  ``training_alignment`` contract with the running policy.

Time is measured from the FILL to the EXIT, in seconds:

    held_seconds = (exit_bar_timestamp - entry_bar_timestamp) - entry_latency_seconds

The entry latency matters: a position is opened at the end of the signal
bar plus the order latency, but it can only be exited when a LATER bar
closes. With 10 s bars and 2 s of latency a one-bar trade lasts 8 s, which
this rule forbids -- the earliest legal exit is then the second bar.
"""
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

#: The requirement itself is the default: absent a ``trade_duration``
#: section, 10 s is enforced.
DEFAULT_MIN_TRADE_SECONDS = 10.0

#: Version of the training-contract block written into model cards.
ALIGNMENT_VERSION = 1

#: The one training population every expert is trained on (see
#: ``src/labeling/harness_labels.py``).
POPULATION_EXPERT_GATE = "expert_gate"

_UNIT_SECONDS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
_GRANULARITY_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd])\s*$", re.IGNORECASE)


class HoldPolicyError(ValueError):
    """The configured hold policy cannot be satisfied; do not trade."""


def parse_granularity_seconds(granularity: str) -> float:
    """``"10s" -> 10.0``, ``"1m" -> 60.0``, ``"5m" -> 300.0`` (any ``<n><s|m|h|d>``)."""
    match = _GRANULARITY_RE.match(str(granularity))
    if not match:
        raise HoldPolicyError(
            f"unrecognised data_granularity {granularity!r}; expected <number><s|m|h|d>, e.g. '10s', '1m'"
        )
    seconds = float(match.group(1)) * _UNIT_SECONDS[match.group(2).lower()]
    if seconds <= 0:
        raise HoldPolicyError(f"data_granularity must be positive, got {granularity!r}")
    return seconds


@dataclass(frozen=True)
class HoldPolicy:
    """No trade may be held shorter than ``min_trade_seconds``."""

    min_trade_seconds: float
    granularity: str
    granularity_seconds: float
    max_holding_bars: int
    entry_latency_seconds: float = 0.0

    # -- construction -------------------------------------------------------

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], entry_latency_seconds: Optional[float] = None,
        validate: bool = True,
    ) -> "HoldPolicy":
        """Build the policy from a merged config.

        ``entry_latency_seconds`` defaults to
        ``trade_duration.assumed_entry_latency_seconds`` and, when that is
        null, to ``execution.simulated_latency_ms / 1000`` (what the paper
        simulator actually charges).
        """
        duration_cfg = config.get("trade_duration", {}) or {}
        granularity = (config.get("system", {}) or {}).get("data_granularity", "1m")
        max_bars = int((config.get("labeling", {}) or {}).get("max_holding_bars", 5))

        if entry_latency_seconds is None:
            assumed = duration_cfg.get("assumed_entry_latency_seconds")
            if assumed is None:
                assumed = float((config.get("execution", {}) or {}).get("simulated_latency_ms", 0)) / 1000.0
            entry_latency_seconds = float(assumed)

        policy = cls(
            min_trade_seconds=float(duration_cfg.get("min_trade_seconds", DEFAULT_MIN_TRADE_SECONDS)),
            granularity=str(granularity),
            granularity_seconds=parse_granularity_seconds(granularity),
            max_holding_bars=max_bars,
            entry_latency_seconds=max(0.0, float(entry_latency_seconds)),
        )
        if validate:
            policy.validate()
        return policy

    def with_latency(self, entry_latency_seconds: float) -> "HoldPolicy":
        """The same policy for a fill that actually took ``entry_latency_seconds``."""
        return HoldPolicy(
            self.min_trade_seconds, self.granularity, self.granularity_seconds,
            self.max_holding_bars, max(0.0, float(entry_latency_seconds)),
        )

    # -- the arithmetic -----------------------------------------------------

    @property
    def min_hold_bars(self) -> int:
        """Fewest bars a position must be held (always at least 1)."""
        if self.min_trade_seconds <= 0:
            return 1
        needed = (self.min_trade_seconds + self.entry_latency_seconds) / self.granularity_seconds
        return max(1, math.ceil(needed - 1e-9))

    def held_seconds(self, bars_held: int) -> float:
        """Wall-clock fill-to-exit time of a position exited after ``bars_held`` bars."""
        return bars_held * self.granularity_seconds - self.entry_latency_seconds

    @property
    def shortest_legal_hold_seconds(self) -> float:
        return self.held_seconds(self.min_hold_bars)

    @property
    def longest_hold_seconds(self) -> float:
        return self.held_seconds(self.max_holding_bars)

    def is_violation(self, held_seconds: float) -> bool:
        """True when a trade of this duration breaks the rule (1 ms tolerance)."""
        return self.min_trade_seconds > 0 and held_seconds < self.min_trade_seconds - 1e-3

    # -- validation ---------------------------------------------------------

    def problems(self) -> List[str]:
        found: List[str] = []
        if self.min_trade_seconds < 0:
            found.append(f"trade_duration.min_trade_seconds must be >= 0, got {self.min_trade_seconds}")
        if self.max_holding_bars < 1:
            found.append(f"labeling.max_holding_bars must be >= 1, got {self.max_holding_bars}")
        elif self.max_holding_bars < self.min_hold_bars:
            found.append(
                f"labeling.max_holding_bars={self.max_holding_bars} x {self.granularity} "
                f"= {self.longest_hold_seconds:g}s held is shorter than the "
                f"{self.min_trade_seconds:g}s minimum (needs >= {self.min_hold_bars} bars): "
                f"every timeout would be an illegal trade. Raise max_holding_bars or lower "
                f"trade_duration.min_trade_seconds."
            )
        if self.entry_latency_seconds >= self.max_holding_bars * self.granularity_seconds > 0:
            found.append(
                f"assumed entry latency {self.entry_latency_seconds:g}s is not shorter than the "
                f"longest hold ({self.max_holding_bars} x {self.granularity})"
            )
        return found

    def validate(self) -> "HoldPolicy":
        problems = self.problems()
        if problems:
            raise HoldPolicyError("; ".join(problems))
        return self

    # -- the contract a model is trained under ------------------------------

    def fingerprint(self) -> Dict[str, Any]:
        """The numbers a model must have been trained under to be served here."""
        return {
            "min_trade_seconds": self.min_trade_seconds,
            "granularity": self.granularity,
            "max_holding_bars": self.max_holding_bars,
            "min_hold_bars": self.min_hold_bars,
            "entry_latency_seconds": round(self.entry_latency_seconds, 6),
        }


#: Components whose models decide (or size, or select) trades, and so must be
#: trained under the hold-policy contract before they can be promoted.
ALIGNED_COMPONENTS = frozenset({
    "trend_expert", "mean_reversion_expert", "breakout_expert",
    "opportunity_filter", "router", "risk_model",
})


def registry_alignment_status(
    metrics: Mapping[str, Any], policy: HoldPolicy, *, expected_population: Optional[str] = None,
) -> "tuple[bool, Optional[str]]":
    """``(ok, problem)`` for a model card / registry metrics dict.

    The retrain cycle records this on every registered version and
    ``approve_model.py`` refuses a version that is not ``ok``.
    """
    problem = model_alignment_problem(
        {"training_alignment": metrics.get("training_alignment")}, policy,
        require_aligned=True, expected_population=expected_population,
    )
    return problem is None, problem


def alignment_settings(config: Mapping[str, Any]) -> "tuple[HoldPolicy, bool]":
    """``(policy, require_aligned_models)`` for a merged config.

    Raises :class:`HoldPolicyError` when the policy is unsatisfiable, so a
    component built from a bad config fails at construction, not per bar.
    ``trade_duration.require_aligned_models`` defaults to False in code
    (components built from a bare config keep working) and is True in the
    shipped ``settings.base.yaml``.
    """
    policy = HoldPolicy.from_config(config)
    require = bool((config.get("trade_duration", {}) or {}).get("require_aligned_models", False))
    return policy, require


# ---------------------------------------------------------------------------
# training contract: written by the training scripts, checked by the loaders
# ---------------------------------------------------------------------------

def training_alignment_block(
    policy: HoldPolicy,
    *,
    population: str,
    sample_policy: str,
    require_opportunity_filter_pass: bool,
    n_candidates: int,
    n_samples: int,
    hold_stats: Optional[Mapping[str, Any]] = None,
    min_duration_violations: int = 0,
) -> Dict[str, Any]:
    """The ``training_alignment`` block stored in a model pickle and its card."""
    return {
        "version": ALIGNMENT_VERSION,
        "hold_policy": policy.fingerprint(),
        "population": population,
        "sample_policy": sample_policy,
        "require_opportunity_filter_pass": bool(require_opportunity_filter_pass),
        "entry_model": "simulated_fill",
        "barrier_sizing": "regime_specific",
        "exit_rule": "barrier_exit_step/v1",
        "n_candidates": int(n_candidates),
        "n_samples": int(n_samples),
        "min_duration_violations": int(min_duration_violations),
        "hold_stats": dict(hold_stats or {}),
    }


def model_alignment_problem(
    payload: Mapping[str, Any],
    policy: Optional[HoldPolicy],
    *,
    require_aligned: bool,
    expected_population: Optional[str] = None,
) -> Optional[str]:
    """Why a loaded model must NOT be served under ``policy`` (None = fine).

    ``policy is None`` disables the check (components constructed without
    a config, e.g. in unit tests). A model with no ``training_alignment``
    block is a LEGACY model: refused when ``require_aligned`` is set,
    because it was trained on labels that do not describe how it is used.
    """
    if policy is None:
        return None
    block = payload.get("training_alignment")
    if block is None:
        if require_aligned:
            return ("model has no training_alignment contract (trained before the minimum-trade-"
                    "duration / harness-alignment change); retrain it")
        return None

    trained = block.get("hold_policy") or {}
    running = policy.fingerprint()
    diffs = [
        f"{key}: trained={trained.get(key)!r} running={running[key]!r}"
        for key in running if trained.get(key) != running[key]
    ]
    if diffs:
        return "hold policy differs from the one it was trained under (" + "; ".join(diffs) + ")"
    if int(block.get("min_duration_violations", 0)) != 0:
        return f"its training labels contained {block['min_duration_violations']} trade(s) shorter than the minimum"
    if expected_population is not None and block.get("population") != expected_population:
        return (f"trained on population {block.get('population')!r}, but is deployed on "
                f"{expected_population!r}")
    return None
