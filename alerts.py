import json
import os
import ssl
import urllib.request

import certifi
from dotenv import load_dotenv

load_dotenv()

# python.org's Python on macOS ships without trusted certificates, so use certifi's
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def send_alert(text):
    """Send an alert to Discord if DISCORD_WEBHOOK_URL is set, otherwise print it.

    Returns True if it reached Discord, False if it fell back to the terminal.
    """
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if url:
        body = json.dumps({"content": text[:2000]}).encode()  # Discord's message limit
        req = urllib.request.Request(
            url,
            data=body,
            # Discord rejects requests without a User-Agent
            headers={"Content-Type": "application/json", "User-Agent": "Lockbox/1.0"},
        )
        try:
            urllib.request.urlopen(req, timeout=10, context=SSL_CONTEXT)
            print(f"[ALERT -> Discord] {text}")
            return True
        except Exception as e:
            # Never print the URL itself: it works as a password for the channel
            print(f"[alert] Discord failed ({type(e).__name__}), printing instead")
    print(f"[ALERT] {text}")
    return False


if __name__ == "__main__":
    send_alert("Lockbox test alert: if you can read this, alerts work.")
