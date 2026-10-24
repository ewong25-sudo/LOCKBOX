import json
import os
import re
import ssl
import urllib.request

import certifi
from dotenv import load_dotenv

load_dotenv()

# python.org's Python on macOS ships without trusted certificates, so use certifi's
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# Card color by the alert's leading emoji (Discord wants an int RGB)
COLORS = {"🚨": 0xE74C3C, "🔁": 0xE67E22, "🤔": 0xF1C40F, "🔒": 0x2ECC71, "⚠️": 0x95A5A6, "🛑": 0x95A5A6}
DEFAULT_COLOR = 0x3498DB


def _discord_payload(text):
    """First line = headline (what the phone notification shows), the rest = a colored card."""
    title, _, body = text.partition("\n")
    payload = {"content": f"**{title}**"[:2000], "username": "Lockbox"}
    user_id = os.getenv("DISCORD_USER_ID", "").strip()
    if user_id.isdigit():
        # @mention the user: Discord notifies mentions even when channel notifications are quiet
        payload["content"] = f"<@{user_id}> {payload['content']}"[:2000]
        payload["allowed_mentions"] = {"users": [user_id]}
    else:
        payload["allowed_mentions"] = {"parse": []}   # never ping @everyone by accident
    if body.strip():
        color = next((c for emoji, c in COLORS.items() if title.startswith(emoji)), DEFAULT_COLOR)
        payload["embeds"] = [{"description": body.strip()[:4096], "color": color,
                              "footer": {"text": "Lockbox · lineup agent"}}]
    return payload


def _plain(text):
    """Terminal version: drop Discord markdown."""
    return re.sub(r"\*\*|```", "", text)


def send_alert(text):
    """Send an alert to Discord if DISCORD_WEBHOOK_URL is set, otherwise print it.

    Format: first line is a short headline, any further lines are the details
    (Discord markdown allowed). Returns True if it reached Discord.
    """
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if url:
        req = urllib.request.Request(
            url,
            data=json.dumps(_discord_payload(text)).encode(),
            # Discord rejects requests without a User-Agent
            headers={"Content-Type": "application/json", "User-Agent": "Lockbox/1.0"},
        )
        try:
            urllib.request.urlopen(req, timeout=10, context=SSL_CONTEXT)
            print(f"[ALERT -> Discord] {_plain(text)}")
            return True
        except Exception as e:
            # Never print the URL itself: it works as a password for the channel
            print(f"[alert] Discord failed ({type(e).__name__}), printing instead")
    print(f"[ALERT] {_plain(text)}")
    return False


if __name__ == "__main__":
    send_alert("✅ Lockbox test alert\nIf you can read this, alerts work.")
