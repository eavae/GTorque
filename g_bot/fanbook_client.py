import websockets
import os
import json
import base64
import asyncio
from aiohttp import ClientSession
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from typing import Optional
from urllib.parse import urlencode
from concurrent.futures import TimeoutError

import websockets.exceptions


class Me(TypedDict):
    id: int
    is_bot: bool
    first_name: str
    last_name: str
    username: str
    avatar: str
    user_token: str
    owner_id: int
    can_join_groups: bool
    can_read_all_group_messages: bool
    supports_inline_queries: bool


class FanBookHttpClient:
    base_url = "https://a1.fanbook.mobi/api"

    def __init__(self):
        self._session = ClientSession()

    async def get_me(self, token: str) -> Me:
        async with self._session.get(f"{self.base_url}/bot/{token}/getMe") as resp:
            resp = await resp.json(content_type=None)
            return resp["result"]

    async def close(self):
        await self._session.close()

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()


class FanBookApp(BaseModel):
    bot_id: int = Field(default_factory=lambda: int(os.getenv("FANBOOK_BOT_ID")))
    bot_token: str = Field(default_factory=lambda: os.getenv("FANBOOK_BOT_TOKEN"))
    socket_url: str = Field(default_factory=lambda: os.getenv("FANBOOK_SOCKET_URL"))

    user_token: Optional[str] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = FanBookHttpClient()

    @property
    def ws_url(self):
        device_id = f"bot{self.bot_id}"
        version = "1.6.60"

        properties = dict(
            device_id=f"bot{self.bot_id}",
            version=version,
            platform="bot",
            channel="office",
            build_number="1",
        )
        properties = base64.b64encode(
            json.dumps(properties).encode("utf8"),
        ).decode("utf8")

        params = {
            "id": self.user_token,
            "dId": device_id,
            "v": version,
            "x-super-properties": properties,
        }

        return f"{self.socket_url}?{urlencode(params)}"

    def on_message(self, message: str):
        msg = json.loads(message)

        if msg["action"] in {"connect", "pong"}:
            return

        # handle other actions
        pass

    async def start(self):
        await self._renew_user_token()

        while True:
            try:
                async with websockets.connect(self.ws_url, open_timeout=8) as socket:
                    try:
                        asyncio.create_task(self._start_heartbeat(socket))

                        while True:
                            message = await socket.recv()
                            self.on_message(message)

                    except websockets.ConnectionClosed:
                        continue
            except websockets.exceptions.InvalidStatusCode as e:
                # 经测试，当出现 502 时，需要重新获取 user_token
                if e.status_code == 502:
                    await self._renew_user_token()
                else:
                    raise e

    async def _renew_user_token(self):
        me = await self._client.get_me(self.bot_token)
        self.user_token = me["user_token"]

    async def _start_heartbeat(self, socket: websockets.WebSocketClientProtocol):
        while True:
            try:
                await asyncio.sleep(25)
                await socket.send('{"type":"ping"}')
            except TimeoutError:
                continue

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self._client.close()


async def main():
    async with FanBookApp() as app:
        await app.start()


if __name__ == "__main__":
    asyncio.run(main())
