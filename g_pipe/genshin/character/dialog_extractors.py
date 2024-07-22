import re
import os
import json
import pandas as pd
from transformers import PreTrainedTokenizer
from typing import Generator, List, Dict, Tuple
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    BaseMessage,
)
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.output_parsers import BaseOutputParser
from collections import defaultdict

from g_core.prompts.dialog import (
    SCENE_DIALOG_CONV_TMPL,
    TOPIC_DIALOG_CONV_TMPL,
    ROLE_PLAY_SYSTEM_TMPL,
    QUESTION_ANSWER_SYSTEM_TMPL,
)
from g_core.prompts.profile import (
    PROFILE_DEFINITION,
    CATALOG_EXTRACTION_TMPL,
    PROFILE_GENERATION_TMPL,
)


ROOT_FOLDER = "../GTorque/g-crawler/knowledge_base"
# chapter = '序章'
# stage = '第一幕'
# scene = '第一场'

# hardcoded exclude
# TODO: using PPL < 100 and PartOfSpeechEntropy > 1.3 to exclude
EXCLUDES = [
    "黑潮与白露的歌剧",
    "终幕礼",
    "独眼小宝总动员",
]
CHINESE_NUMBERS = "一二三四五六七八九十"


class DialogOutput(BaseModel):
    summary: str
    dialog: str


class SceneDialogOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        summary = ""
        dialog = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("情节："):
                summary = line.replace("情节：", "").strip()
            elif len(line) > 0:
                dialog.append(line)
        return DialogOutput(summary=summary, dialog="\n".join(dialog))


class MQAOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        conversations = []
        question: str = ""
        answer: str = ""
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("Q:"):
                question = line.replace("Q:", "").strip()
            elif line.startswith("A:"):
                answer = line.replace("A:", "").strip()

            if question and answer:
                conversations.append((question, answer))
                question = ""
                answer = ""
        return conversations

    def get_format_instructions(self):
        return (
            "输出时，一个问题或回答一行，问题与回答交替。比如：\n"
            "Q: 问题1\n"
            "A: 回答1\n"
            "Q: 问题2\n"
            "A: 回答2\n"
            "...(可以有多个问题和回答)\n"
        )


class LengthAwareChatSplitter:
    """
    Split the chat into multiple parts based on the length of the chat
    """

    from llamafactory.data.template import TEMPLATES

    def __init__(
        self,
        template: str,
        tokenizer: "PreTrainedTokenizer",
        max_tokens: int = 4096,
        reserved_label_len: int = 1,
    ):
        if template not in self.TEMPLATES:
            raise ValueError(f"Template {template} not found")
        self._tokenizer = tokenizer
        self._template = self.TEMPLATES[template]
        self._max_tokens = max_tokens
        self._reserved_label_len = reserved_label_len

    def _get_sub_parts(
        self,
        messages: List[BaseMessage],
        system_message: SystemMessage = None,
    ) -> Tuple[List[BaseMessage], List[BaseMessage]]:
        if len(messages) == 0:
            return []

        input_ids = []
        for i, message in enumerate(messages):
            elements = []
            if i == 0 and system_message:
                elements += self._template.format_system.apply(
                    content=system_message.content
                )

            elif i > 0 and i % 2 == 0:
                elements += self._template.format_separator.apply()

            if isinstance(message, HumanMessage):
                elements += self._template.format_user.apply(
                    content=message.content, idx=str(i // 2)
                )
            elif isinstance(message, AIMessage):
                elements += self._template.format_assistant.apply(
                    content=message.content
                )
            elif isinstance(message, ToolMessage):
                elements += self._template.format_observation.apply(
                    content=message.content
                )
            elif isinstance(message, ToolCall):
                elements += self._template.format_function.apply(
                    content=json.dumps(
                        {
                            "name": message.name,
                            "arguments": message.args,
                        }
                    )
                )
            else:
                raise NotImplementedError(
                    "Unexpected Message Type: {}".format(message.__class__)
                )

            input_ids.extend(
                self._template._convert_elements_to_ids(self._tokenizer, elements)
            )

            # check if the input_ids exceed the max_tokens
            if len(input_ids) > self._max_tokens:
                # find the last ai message
                ai_message_idx = -1
                for j in range(i - 1, 0, -1):
                    if isinstance(messages[j], AIMessage):
                        ai_message_idx = j
                        break

                # raise warning if no ai message found
                if ai_message_idx == -1:
                    raise Warning(
                        "Cannot split the chat into multiple parts, no AI message found"
                    )

                # split the messages
                ai_message_idx = ai_message_idx + 1
                return (
                    [system_message] + messages[:ai_message_idx],
                    messages[ai_message_idx:],
                )

        return [system_message] + messages, []

    def split(self, chat: ChatPromptValue) -> Generator[ChatPromptValue, None, None]:
        messages = chat.to_messages()
        system_message = None
        if messages and isinstance(messages[0], SystemMessage):
            system_message = messages[0]
            messages = messages[1:]

        while messages:
            sub_parts, messages = self._get_sub_parts(messages, system_message)
            if sub_parts:
                yield ChatPromptValue(messages=sub_parts)


def chinese2arabic(s: str) -> int:
    if not re.match(r".*[一二三四五六七八九十]+.*", s):
        return 1

    s = re.match(r".*([一二三四五六七八九十]+).*", s).group(1)
    for i, n in enumerate(CHINESE_NUMBERS):
        s = s.replace(n, str(i + 1))
    return int(s)


def extract_scenes(s: str):
    scenes = []
    for line in s.split("\n"):
        # 序章 第二幕为了没有眼泪的明天
        segments = line.split(" ")
        if len(segments) >= 2:
            is_end_with_chapter = segments[0].endswith("章")
            is_start_with_n_stage = re.match(
                r"第[一二三四五六七八九十]+幕", segments[1]
            )
            if is_end_with_chapter and is_start_with_n_stage:
                chapter_name = segments[0]

                scene = segments[1]
                index_of_stage = scene.index("幕") + 1
                scene_name = scene[index_of_stage:]
                scene_order = scene[:index_of_stage]

                scenes.append(
                    {
                        "chapter": chapter_name,
                        "chapter_no": chinese2arabic(chapter_name),
                        "scene_name": scene_name,
                        "scene_no": chinese2arabic(scene_order),
                    }
                )
    return scenes


def extract_acts_from_scenes(scenes: List[str]):
    # 详细对话内容，请查阅词条「水的女儿」
    # match 「水的女儿」
    acts = []
    for scene in scenes:
        scene_name = scene["scene_name"]

        # check existence
        file_path = f"{ROOT_FOLDER}/原神/{scene_name}.md"
        if not os.path.exists(file_path):
            print(f"{file_path} not exists")
            continue

        # check excludes
        if scene_name in EXCLUDES:
            print(f"{scene_name} is excluded")
            continue

        # extract sub stages
        file_content = open(
            f"{ROOT_FOLDER}/原神/{scene_name}.md", "r", encoding="utf-8"
        ).read()

        new_acts = []
        act_index = 0
        for line in file_content.split("\n"):
            if re.match(r".*对话.*词条(.*)", line):
                act_index += 1
                act_name = re.match(r".*对话.*词条(.*)", line).groups()[0]

                # check act exists
                if act_name in EXCLUDES:
                    print(f"{act_name} is excluded")
                    continue
                file_path = f"{ROOT_FOLDER}/原神/{act_name}.md"
                if not os.path.exists(file_path):
                    print(f"{file_path} not exists")
                    continue

                temp_act = deepcopy(scene)
                temp_act["act_name"] = act_name
                temp_act["act_no"] = act_index
                new_acts.append(temp_act)

        # if no sub stages
        if new_acts:
            acts.extend(new_acts)
        else:
            scene["act_name"] = scene["scene_name"]
            scene["act_no"] = 1
            acts.append(scene)

    return acts


def convert_choice_to_dialog(s: str):
    main_character = "旅行者"
    main_message = ""
    response = ""

    # get main message
    for line in s.split("\n"):
        if line.startswith("旅行者："):
            main_message = line
            break

        if line.strip() and not line.startswith("★"):
            main_message = f"{main_character}：{line}"
            break

    # get reply
    for line in reversed(s.split("\n")):
        line = line.strip()

        if "：" in line:
            response = line

    if response:
        return f"* {main_message}\n* {response}"
    return f"* {main_message}"


def format_dialog(stages: List[Dict]):
    for stage in tqdm(stages, desc="Cleaning"):
        chapter = stage["chapter"]
        scene = stage["scene_name"]
        act = stage["act_name"]

        folder_path = f"data/dialog/{chapter}/{scene}"
        file_path = f"{folder_path}/{act}.md"
        file_content = open(file_path, "r", encoding="utf-8").read()
        matches = re.search(r"(\n★\n[\s\S]+?\n\n)\*", file_content, re.MULTILINE)

        while matches:
            dialog = matches.group(1)
            dialog = convert_choice_to_dialog(dialog)
            print(dialog)

            # replace dialog
            file_content = (
                file_content[: matches.start(1)]
                + f"{dialog}\n"
                + file_content[matches.end(1) :]  # noqa
            )

            matches = re.search(r"(\n★\n[\s\S]+?\n\n)\*", file_content, re.MULTILINE)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_content)


def _filter_fragment(fragment: str):
    if fragment.startswith("首页"):
        return False
    if fragment.startswith("目录"):
        return False
    return True


def extract_dialog_with_llm(
    stages: List[Dict],
    from_root: str = "data/dialog_text",
    to_root: str = "data/dialog_structured",
):
    # check to_root folder exists or not
    if not os.path.exists(to_root):
        os.makedirs(to_root)

    llm = ChatOpenAI(
        model="deepseek-chat", max_retries=3, max_tokens=4096, temperature=0.01
    )
    template = PromptTemplate.from_template(SCENE_DIALOG_CONV_TMPL)
    chain = template | llm | SceneDialogOutputParser()

    # group by chapter
    df = pd.DataFrame(stages)
    for chapter, group in tqdm(df.groupby("chapter"), desc="Processing Chapters"):
        pre_summary = ""
        group_df = group.sort_values(by=["scene_no", "act_no"])
        for row_i in trange(len(group_df), desc="Processing Acts"):
            row = group_df.iloc[row_i]
            chapter = row["chapter"]
            scene = row["scene_name"]
            act = row["act_name"]

            folder_path = f"{from_root}/{chapter}/{scene}"
            file_path = f"{folder_path}/{act}.md"
            file_content = open(file_path, "r", encoding="utf-8").read()

            save_to_folder = f"{to_root}/{chapter}_{act}"
            if not os.path.exists(save_to_folder):
                os.makedirs(save_to_folder)

            fragments = file_content.split("\n## ")
            fragments = list(filter(_filter_fragment, fragments))
            fragments = [y for x in fragments for y in x.split("\n### ")[1:]]
            for i, fragment in tenumerate(fragments, desc="Processing Fragments"):
                save_to = f"{save_to_folder}/{i}.json"

                # check if file exists
                if os.path.exists(save_to):
                    # load pre_summary
                    pre_summary = json.load(open(save_to, "r"))["summary"]
                    continue

                response: DialogOutput = chain.invoke(
                    {
                        "pre_summary": pre_summary,
                        "context": fragment,
                    }
                )
                pre_summary = response.summary

                data = response.model_dump()
                data.update(row.to_dict())
                with open(save_to, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)


def stats_all_character(root_folder="data/dialog_structured"):
    all_characters = defaultdict(int)
    for file_path in Path(root_folder).rglob("*.json"):
        data = json.load(open(file_path, "r"))
        dialog = data["dialog"]
        for line in dialog.split("\n"):
            if "：" in line:
                character = line.split("：")[0]
                all_characters[character] += 1
    return all_characters


def get_topic_conversion_chain():
    llm = ChatOpenAI(
        model="deepseek-chat", max_retries=3, max_tokens=4096, temperature=0.01
    )
    parser = MQAOutputParser()
    prompt = PromptTemplate.from_template(
        TOPIC_DIALOG_CONV_TMPL,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser


def convert_topic_to_dialog(text_content: str, llm_chain):
    text = text_content.split("文本语言：")[1].strip()
    text_lines = text.split("\n\n")
    chunk_size = len(text_lines) // 2
    dialogs = []
    for i in range(2):
        chunk = text_lines[i * chunk_size : (i + 1) * chunk_size]  # noqa
        chunk_text = "\n".join(chunk)
        qa_pairs = llm_chain.invoke({"context": chunk_text})
        dialogs.extend(qa_pairs)
    return dialogs


def extract_markdown_catalog(text_content: str):
    catalogs = []
    for line in text_content.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            catalogs.append(line)
            continue

        if len(catalogs) > 0 and line and catalogs[-1].startswith("#"):
            catalogs.append(line)
            catalogs.append("...\n")

    return "\n".join(catalogs)


def filter_markdown_by_catalogs(text_content: str, catalogs: List[str]):
    fragments = []
    continue_with = ""
    for line in text_content.split("\n"):
        line = line.strip()
        if line in catalogs:
            fragments.append(line)
            continue_with = line.split(" ")[0]
            continue

        if line.startswith(continue_with):
            continue_with = ""
            continue

        if continue_with:
            fragments.append(line)

    return "\n".join(fragments)


def get_catalog_extraction_chain():
    llm = ChatOpenAI(
        model="deepseek-chat", max_retries=3, max_tokens=4096, temperature=0.1
    )
    prompt = PromptTemplate.from_template(
        CATALOG_EXTRACTION_TMPL,
        partial_variables={"profile": PROFILE_DEFINITION},
    )
    return prompt | llm


def get_profile_chain():
    llm = ChatOpenAI(
        model="deepseek-chat", max_retries=3, max_tokens=4096, temperature=0.1
    )
    prompt = PromptTemplate.from_template(
        PROFILE_GENERATION_TMPL,
        partial_variables={"profile": PROFILE_DEFINITION},
    )
    return prompt | llm


def dialog_to_conversations(dialog: str, character: str):
    """
    将以`：`分隔的对话转换为对话列表，并且：
    1. 将多行对话合并为一行
    2. 以指定 character 开头的对话作为响应，其它作为查询
    """
    character = f"{character}："
    conversations = []

    queries = []
    responses = []
    for line in dialog.split("\n"):
        if line.startswith(character):
            responses.append(line.replace(character, "").strip())
        else:
            if queries and responses:
                conversations.append(("\n".join(queries), "".join(responses)))
                queries = []
                responses = []

            if queries:
                prev_speaker = queries[-1].split("：")[0] + "："
                if line.startswith(prev_speaker):
                    line = line.replace(prev_speaker, "")
                    queries[-1] += line
                else:
                    queries.append(line)
            else:
                queries.append(line)

    if queries and responses:
        conversations.append(("\n".join(queries), "".join(responses)))

    return conversations


def iter_numbered_files(root_folder: str, suffix: str = ".md"):
    """
    按数字顺序遍历文件夹中的文件
    """
    file_paths = set(str(x) for x in Path(root_folder).rglob(f"*{suffix}"))
    i = 0
    while f"{root_folder}/{i}{suffix}" in file_paths:
        yield f"{root_folder}/{i}{suffix}"
        i += 1


def iter_numbered_dialog(root_folder: str, for_character: str = None):
    """
    遍历文件夹中的对话，并且合并相同的summary
    当指定角色时，只返回包含该角色的对话
    """
    summary = ""
    dialogs = []
    for file_path in iter_numbered_files(root_folder, suffix=".json"):
        data = json.load(open(file_path, "r"))

        if for_character and f"{for_character}：" not in data["dialog"]:
            continue

        if summary == data["summary"]:
            dialogs.append(data["dialog"])
        elif summary:
            yield summary, "\n".join(dialogs)
            summary = data["summary"]
            dialogs = [data["dialog"]]
        else:
            summary = data["summary"]
            dialogs = [data["dialog"]]

    if summary:
        yield summary, "\n".join(dialogs)


class ChatType:
    ROLE_PLAY = "role_play"  # 情节对话
    QUESTION_ANSWER = "question_answer"  # 问答对话


ROLE_PLAY_PROMPT = ChatPromptTemplate.from_messages(
    messages=[
        SystemMessagePromptTemplate.from_template(ROLE_PLAY_SYSTEM_TMPL),
    ]
)
QUESTION_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    messages=[
        SystemMessagePromptTemplate.from_template(QUESTION_ANSWER_SYSTEM_TMPL),
    ]
)


def to_prompt_value(
    chat_type: ChatType, file_path: str, profile: str
) -> ChatPromptValue:
    """
    将对话转换为 OpenAI 格式
    """
    if chat_type == ChatType.ROLE_PLAY:
        # 读取文件
        data = json.load(open(file_path, "r"))
        summary = data["summary"]
        conversations = data["conversations"]

        # 转换为对话列表
        prompt_value = ROLE_PLAY_PROMPT.invoke({"profile": profile, "context": summary})
        for query, response in conversations:
            prompt_value.messages.append(HumanMessage(content=query))
            prompt_value.messages.append(AIMessage(content=response))

        return prompt_value

    if chat_type == ChatType.QUESTION_ANSWER:
        # 读取文件
        conversations = json.load(open(file_path, "r"))

        # 转换为对话列表
        prompt_value = QUESTION_ANSWER_PROMPT.invoke({"profile": profile})
        for query, response in conversations:
            prompt_value.messages.append(HumanMessage(content=query))
            prompt_value.messages.append(AIMessage(content=response))

        return prompt_value


def langchain_chat_value_to_xtuner_format(chat: ChatPromptValue):
    """
    将 langchain 的对话格式转换为 xtuner 的格式
    """
    messages = chat.to_messages()
    conversations = []
    one_turn = dict()
    for message in messages:
        if isinstance(message, HumanMessage):
            one_turn["input"] = message.content
        elif isinstance(message, AIMessage):
            one_turn["output"] = message.content
        elif isinstance(message, SystemMessage):
            one_turn["system"] = message.content

        if one_turn.get("input", None) and one_turn.get("output", None):
            conversations.append(one_turn)
            one_turn = dict()
    return conversations


if __name__ == "__main__":
    from transformers import AutoTokenizer

    PROFILE_ROOT = "data/characters/genshin/profiles"
    TO_ROOT = "data/characters/genshin/xtuner"
    ROLE_PLAY_ROOT = "data/characters/genshin/conversations"
    QA_ROOT = "data/characters/genshin/qa"

    # create save to folder if not exists
    # if not os.path.exists(SAVE_TO):
    #     os.makedirs(SAVE_TO)
    splitter = LengthAwareChatSplitter(
        template="qwen",
        tokenizer=AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat"),
        max_tokens=4096 * 2,
    )

    file_paths = list(Path(PROFILE_ROOT).rglob("*.md"))
    for file_path in tqdm(file_paths, desc="Processing Files"):
        character_name = file_path.name.replace(".md", "")
        with open(file_path, "r", encoding="utf-8") as f:
            profile = f.read()

        # group all the files into a tuple
        chat_files = []
        for chat_file_path in Path(ROLE_PLAY_ROOT).rglob(f"{character_name}/*.json"):
            # check chat_file_path exists
            if not os.path.exists(chat_file_path):
                continue
            chat_files.append((ChatType.ROLE_PLAY, str(chat_file_path)))
        character_qa_file_path = f"{QA_ROOT}/{character_name}.json"
        if os.path.exists(character_qa_file_path):
            chat_files.append((ChatType.QUESTION_ANSWER, character_qa_file_path))

        chat_objects = [
            to_prompt_value(chat_type, file_path, profile)
            for chat_type, file_path in chat_files
        ]
        _chats = []
        for chat in chat_objects:
            _chats.extend(splitter.split(chat))

        # save to jsonl
        save_to = f"{TO_ROOT}/{character_name}.jsonl"
        with open(save_to, "w", encoding="utf-8") as f:
            for i, chat in enumerate(_chats):
                conversations = langchain_chat_value_to_xtuner_format(chat)
                f.write(json.dumps(conversations, ensure_ascii=False) + "\n")
        pass
