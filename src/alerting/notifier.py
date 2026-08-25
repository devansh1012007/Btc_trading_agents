"""
Alerting (Phase 4, §12): a single notify(event_type, message) call wired
into every place that would want it (circuit breakers, the approval
queue, execution errors). Off by default (alerting.enabled: false) --
enabling this later is a config flip and a bot token, not a refactor.
Only a Telegram backend is implemented (Project_Plan.md's own "Future
Upgrades" list names it); the interface is generic enough to add others
without touching call sites.
"""
import logging
import os

logger = logging.getLogger(__name__)


class Notifier:
    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)
        self.backend = config.get("backend", "telegram")
        self.notify_on = set(config.get("notify_on", []))
        self._token_env_var = config.get("telegram_bot_token_env_var", "BTC_BOT_TELEGRAM_TOKEN")
        self._chat_id_env_var = config.get("telegram_chat_id_env_var", "BTC_BOT_TELEGRAM_CHAT_ID")

    def notify(self, event_type: str, message: str) -> bool:
        """Returns True if a notification was actually sent (or would
        have been, in a dry-run/disabled state this always returns
        False without raising -- alerting failures should never take
        down the trading loop)."""
        if not self.enabled:
            logger.debug(f"[alerting disabled] {event_type}: {message}")
            return False
        if self.notify_on and event_type not in self.notify_on:
            return False

        try:
            if self.backend == "telegram":
                return self._send_telegram(event_type, message)
            logger.warning(f"Unknown alerting backend '{self.backend}', dropping notification")
            return False
        except Exception as e:
            # Alerting must never crash the caller -- a notification
            # failure is not a reason to halt trading.
            logger.error(f"Alert send failed ({event_type}): {e}")
            return False

    def _send_telegram(self, event_type: str, message: str) -> bool:
        token = os.environ.get(self._token_env_var)
        chat_id = os.environ.get(self._chat_id_env_var)
        if not token or not chat_id:
            logger.warning(
                f"Telegram alerting enabled but {self._token_env_var}/{self._chat_id_env_var} "
                "not set in the environment -- dropping notification."
            )
            return False

        import urllib.parse
        import urllib.request

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = urllib.parse.urlencode({"chat_id": chat_id, "text": f"[{event_type}] {message}"}).encode()
        req = urllib.request.Request(url, data=payload)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
