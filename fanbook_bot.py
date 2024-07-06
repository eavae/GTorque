import base64
import json
import threading
import time
import requests
import asyncio
import websockets
import os

from g_core.chat_engine import buildChatEngine

## 环境变量读入
# 在环境变量中设置自己的bot token和id
TOKEN = os.environ.get("BOT_TOKEN", "")
BOT_ID = os.environ.get("BOT_ID", "")
BASE_URL = os.environ.get("BASE_URL", "")

async def on_message(message):
    s = message.decode('utf8')
    obj = json.loads(s)
    action = obj["action"]
    if action == "push":
        content = json.loads(obj["data"]["content"])
        ct = content["type"]
        user_id = obj["data"]["user_id"]
        channel_id = obj["data"]["channel_id"]
        if ct == "text":
            text = content["text"]
            print("==> get text message, try to reply for msg: " + text)
            # 目前先统一用test，后期根据不同的社区使用不同的db
            chat_engine = buildChatEngine("test","user_id")
            response = await chat_engine.achat(text)
            print("reply text: " + response.response)
            await sendMsg(user_id, channel_id, "")
        else:
            print(f"content-type: {ct} is not currently subscribed")
    else:
        print(f"action {action} is not currently subscribed")

async def sendMsg(user_id, channel_id, msg):
    url = f"{BASE_URL}/bot/{TOKEN}/sendMessage"
    msg = "${@!" + str(user_id) + "}" + msg
    print(f"send {msg} to {user_id} in channel {channel_id}")
    data = {
        "chat_id": int(channel_id),
        "text": "{\"type\":\"richText\",\"title\":\"机器人自动回复\",\"document\":\"[{\\\"insert\\\":\\\"”+ msg+ “\\\"}]\"}",
        "parse_mode": "Fanbook","desc": "123", "users": ["all"]
    }
    res = requests.post(url, data=data)
    print("send res: ")
    print(res.json())
def send_ping(ws):
    while True:
        time.sleep(20)
        asyncio.run(ws.send('{"type":"ping"}'))

def get_me():
    response = requests.get(f"{BASE_URL}/bot/{TOKEN}/getMe", timeout=3)
    return response.json()


async def handleWS(user_token):
    version = '1.6.60'
    device_id = f'bot{BOT_ID}'
    header_map = json.dumps({
        "device_id": device_id,
        "version": version,
        "platform": "bot",
        "channel": "office",
        "build_number": "1"
    })
    super_str = base64.b64encode(header_map.encode('utf8')).decode('utf8')
    addr = f'wss://gateway-bot.fanbook.mobi/websocket?id={user_token}&dId={device_id}&v={version}&x-super-properties={super_str}'
    async with websockets.connect(addr) as ws:
        # asyncio.run(send_ping(ws))
        ping_thread = threading.Thread(target=send_ping, args=(ws,))
        ping_thread.daemon = True
        ping_thread.start()
        while True:
            evt_data = await ws.recv()
            print("==> receive data: {0}".format(evt_data))
            try:
                await on_message(evt_data)
            except Exception as e:
                print("WebSocketError: ", e)


if __name__ == '__main__':
    res = get_me()
    user_token = res["result"]['user_token']
    asyncio.run(handleWS(user_token))
