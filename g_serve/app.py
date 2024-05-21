import os
import asyncio
from typing import Union
from fastapi import FastAPI
from contextlib import asynccontextmanager
from g_serve.bot.client import client


@asynccontextmanager
async def lifespan(app: FastAPI):
    coroutine = client.start(
        appid=os.getenv("G_BOT_APPID"),
        secret=os.getenv("G_BOT_APPSECRET"),
        ret_coro=True,
    )
    asyncio.create_task(coroutine)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/{app_id}.json")
def bot_verify(app_id: str):
    v_app_id = os.getenv("G_BOT_APPID")
    if app_id == v_app_id:
        return {
            "bot_appid": v_app_id,
        }


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("G_SERVE_HOST", "0.0.0.0"),
        port=int(os.getenv("G_SERVE_PORT", 8000)),
    )
