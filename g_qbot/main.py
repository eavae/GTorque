import os
import botpy
import g_core.setup_globals  # noqa
from g_qbot.client import BotClient


if __name__ == "__main__":
    client = BotClient(
        intents=botpy.Intents(direct_message=True),
        is_sandbox=True if os.getenv("G_BOT_SANDBOX") == "1" else False,
    )

    client.run(
        appid=os.getenv("G_BOT_APPID"),
        secret=os.getenv("G_BOT_APPSECRET"),
    )
