import botpy
from botpy.message import Message, DirectMessage


class BotClient(botpy.Client):
    async def on_at_message_create(self, message: Message):
        await message.reply(
            content=f"机器人{self.robot.name}收到你的@消息了: {message.content}"
        )

    async def on_direct_message_create(self, message: DirectMessage):
        """
        此处为处理该事件的代码
        """
        await message.reply(
            content=f"机器人{self.robot.name}收到你的私信: {message.content}"
        )

    async def on_direct_message_delete(self, message: DirectMessage):
        """
        此处为处理该事件的代码
        """
        await message.reply(
            content=f"机器人{self.robot.name}收到你的撤回请求: {message.content}"
        )
