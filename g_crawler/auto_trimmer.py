import re
import os
from typing_extensions import TypedDict
from typing import List, Optional
from enum import Enum
from langgraph.graph import StateGraph
from langchain_core.output_parsers import BaseOutputParser
from tokenizers import Tokenizer
from typing import Literal
from lxml.html import HTMLParser, HtmlElement
from pydantic import BaseModel
from collections import defaultdict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from g_crawler.etree_tools import extract_left_subtree, decode_url, to_string
from g_core.prompts.xpath import (
    HTML_PRUNING_TMPL,
    HTML_PRUNING_RULE_TESTING_TMPL,
    HTML_PRUNING_FIX_TMPL,
)

html_parser = HTMLParser(
    remove_comments=True,
    remove_blank_text=True,
)


class InlineCodeCheckingOutput(BaseModel):
    passed: bool
    code: Optional[str] = None


class CheckingWithInlineCodeOutputParser(BaseOutputParser):
    rejected_message = "无需移除"

    def parse(self, text: str) -> str:
        passed = True
        inline_code = None

        lines = text.split("\n")
        for line in reversed(lines):
            # 提取 inline 代码块``
            matches = re.findall(r"`([^`]+)`", line)
            if not inline_code and matches:
                inline_code = matches[0]
            if self.rejected_message in line:
                passed = False

        return InlineCodeCheckingOutput(passed=passed, code=inline_code)


class QAOutput(BaseModel):
    is_pass: bool
    feedback: Optional[str] = None


class QuantityAssuranceOutputParser(BaseOutputParser):
    answer_prefix: str = "结论："
    checking: str = "通过"
    no_checking: str = "不通过"
    feedback: str = "修改建议："

    def parse(self, text: str) -> str:
        if self.feedback in text:
            feedback = text.split(self.feedback)[-1].strip()
        else:
            feedback = None

        if self.no_checking in text:
            return QAOutput(is_pass=False, feedback=feedback)

        return QAOutput(is_pass=True, feedback=feedback)


def get_extract_xpath_for_pruning_chain():
    parser = CheckingWithInlineCodeOutputParser()
    llm = ChatOpenAI(
        model=os.getenv("G_CRAWLER_EXTRACT_XPATH_LLM"),
        temperature=0.1,
        max_tokens=1024,
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=HTML_PRUNING_TMPL),
            HumanMessagePromptTemplate.from_template(
                "标题：{title}\n```html\n{content}\n```"
            ),
        ]
    )
    return prompt_template | llm | parser


def get_fix_xpath_for_pruning_chain():
    parser = CheckingWithInlineCodeOutputParser()
    llm = ChatOpenAI(
        model=os.getenv("G_CRAWLER_EXTRACT_XPATH_LLM"),
        temperature=0.1,
        max_tokens=1024,
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=HTML_PRUNING_FIX_TMPL),
            HumanMessagePromptTemplate.from_template(
                "标题：{title}\n```html\n{content}\n```\n\n\n历史反馈（请认真考虑反馈，禁止再次提出相同的XPath）：\n{feedbacks}"
            ),
        ]
    )
    return prompt_template | llm | parser


def get_test_xpath_for_pruning_chain():
    parser = QuantityAssuranceOutputParser()
    llm = ChatOpenAI(
        model=os.getenv("G_CRAWLER_TEST_XPATH_LLM"),
        temperature=0.1,
        max_tokens=1024,
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=HTML_PRUNING_RULE_TESTING_TMPL),
            HumanMessagePromptTemplate.from_template("{context}"),
        ]
    )
    return prompt_template | llm | parser


# 若某个规则被 reject 了，则使用 reject_reason 重新创建规则
class RuleStates(str, Enum):
    PLANED = "planed"  # 未创建规则
    CREATED = "created"  # planed -> created | exhausted 创建规则
    HARD_VALIDATED = (
        "hard_validated"  # created -> hard_validated | rejected 是否选择到元素
    )
    SOFT_VALIDATED = "soft_validated"  # hard_validated -> soft_validated | rejected | accepted 评估选择到的内容是否都是无关的
    REJECTED = "rejected"
    ACCEPTED = "accepted"
    EXHAUSTED = "exhausted"  # 无需继续剪枝


class HitFragment(TypedDict):
    draft_id: int
    content: str


class RejectReason(TypedDict):
    rule: str
    reason: str


class RuleState(TypedDict):
    state: RuleStates
    draft_id: int
    rule: Optional[str]
    reject_reasons: List[RejectReason] = []
    hit_fragments: List[HitFragment] = []


class Draft(TypedDict):
    tree: HtmlElement
    title: str


class State(TypedDict):
    drafts: List[Draft] = []
    rules: List[RuleState] = []


def plan_rule(state: State):
    draft_ids = list(range(len(state["drafts"])))

    # 剔除 exhausted 的规则
    for rule in state["rules"]:
        if rule["state"] == RuleStates.EXHAUSTED:
            draft_ids.remove(rule["draft_id"])

    # 剔除 text content 为空的 draft
    for idx, draft in enumerate(state["drafts"]):
        if not draft["tree"].text_content().strip():
            draft_ids.remove(idx)

    # 根据使用频率倒序排列
    rule_counts = defaultdict(int)
    for rule in state["rules"]:
        rule_counts[rule["draft_id"]] += 1
    draft_ids = sorted(draft_ids, key=lambda x: rule_counts[x])

    # 派发任务
    state["rules"].append(
        {
            "state": RuleStates.PLANED,
            "draft_id": draft_ids[0],
            "rule": None,
            "reject_reasons": [],
        }
    )


def apply_rule(state: State, rule: RuleState):
    """应用规则时，将规则应用到所有的 draft 上"""
    for draft in state["drafts"]:
        tree = draft["tree"]
        for element in tree.xpath(rule["rule"]):
            element.getparent().remove(element)


def exhausted(state: State):
    if len(state["rules"]) >= 3:
        exhausted_count = 0
        for rule in state["rules"][-3:]:
            if rule["state"] == RuleStates.EXHAUSTED:
                exhausted_count += 1
        if exhausted_count == 3:
            return True
    return False


def supervisor(state: State):
    # 如果最后三个规则都是 exhausted，则结束
    if exhausted(state):
        return state

    # 派发任务
    plan_rule(state)

    return state


x_path_pruning_chain = get_extract_xpath_for_pruning_chain()
x_path_fix_chain = get_fix_xpath_for_pruning_chain()
tokenizer = Tokenizer.from_pretrained("deepseek-ai/DeepSeek-V2")


def rule_creator(state: State):
    assert len(state["rules"]) > 0, "No rule planned to create"

    rule_state = state["rules"][-1]
    assert rule_state["state"] in {
        RuleStates.PLANED,
        RuleStates.REJECTED,
    }, "Only planed or rejected rule can be created"

    draft = state["drafts"][rule_state["draft_id"]]
    left_subtree = extract_left_subtree(
        tokenizer=tokenizer,
        element=draft["tree"],
        max_tokens=2048,
    )
    left_subtree = decode_url(left_subtree)

    # create a new rule
    if rule_state["state"] == RuleStates.PLANED:
        xpath: InlineCodeCheckingOutput = x_path_pruning_chain.invoke(
            {
                "title": draft["title"],
                "content": to_string(left_subtree, pretty_print=True),
            }
        )
    # fix existing rule by feedback, it's only works when the rule is not passed by hard validator
    elif rule_state["state"] == RuleStates.REJECTED:
        feedbacks = []
        for idx, reason in enumerate(rule_state["reject_reasons"]):
            feedbacks.append(f"{idx}. `{reason['rule']}`: {reason['reason']}")
        feedbacks_str = "\n".join(feedbacks)

        xpath: InlineCodeCheckingOutput = x_path_fix_chain.invoke(
            {
                "feedbacks": feedbacks_str,
                "title": draft["title"],
                "content": to_string(left_subtree, pretty_print=True),
            }
        )

    if xpath.passed:
        # 如果继续提出相同的规则，则标记为 exhausted
        if rule_state["rule"] == xpath.code:
            rule_state["state"] = RuleStates.EXHAUSTED
            print(f"Rule exhausted: {rule_state['rule']} by repeated rule.")
        else:
            rule_state["state"] = RuleStates.CREATED
            rule_state["rule"] = xpath.code
    else:
        rule_state["state"] = RuleStates.EXHAUSTED
        rule_state["rule"] = None

    return state


def rule_hard_validator(state: State):
    assert (
        len(state["rules"]) > 0 and state["rules"][-1]["state"] == RuleStates.CREATED
    ), "No created rule founded"

    rule_state = state["rules"][-1]

    fragments = []
    for idx, draft in enumerate(state["drafts"]):
        tree = draft["tree"]
        for element in tree.xpath(rule_state["rule"]):
            fragments.append(
                {"draft_id": idx, "content": to_string(element, pretty_print=True)}
            )

    if len(fragments) == 0:
        rule_state["state"] = RuleStates.REJECTED
        rule_state["reject_reasons"].append(
            dict(
                rule=rule_state["rule"],
                reason=f"该 XPath 在{len(state['drafts'])}篇文章中未命中任何元素，请修改。",
            )
        )
    else:
        rule_state["state"] = RuleStates.HARD_VALIDATED
        rule_state["hit_fragments"] = fragments

    return state


def format_hit_fragments(fragments: List[HitFragment], drafts: List[Draft]) -> str:
    result = []
    for fragment in fragments:
        parts = []
        draft_id = fragment["draft_id"]
        draft = drafts[draft_id]
        parts.append(f"文章标题：{draft['title']}")
        parts.append("```html")
        parts.append(fragment["content"])
        parts.append("```")
        result.append("\n".join(parts))
    return "\n\n".join(result)


soft_validator_chain = get_test_xpath_for_pruning_chain()


def rule_soft_validator(state: State):
    assert (
        len(state["rules"]) > 0
        and state["rules"][-1]["state"] == RuleStates.HARD_VALIDATED
    ), "No hard validated rule founded"

    rule_state = state["rules"][-1]
    fragments_str = format_hit_fragments(
        rule_state["hit_fragments"][:5], state["drafts"]
    )
    context = f"XPath: {rule_state['rule']}\n\n{fragments_str}"
    qa_output: QAOutput = soft_validator_chain.invoke({"context": context})

    if qa_output.is_pass:
        rule_state["state"] = RuleStates.ACCEPTED
        apply_rule(state, rule=rule_state)
        print(f"Rule accepted: {rule_state['rule']}")
    else:
        rule_state["state"] = RuleStates.EXHAUSTED
        rule_state["reject_reasons"].append(
            dict(rule=rule_state["rule"], reason=qa_output.feedback)
        )
        print(f"Rule exhausted: {rule_state['rule']} by qa failed.")

    return state


def route_from_supervisor(state: State) -> Literal["rule_creator", "__end__"]:
    # if no rules, throw error
    if len(state["rules"]) == 0:
        raise ValueError("No rules to route")

    if exhausted(state):
        return "__end__"

    last_rule = state["rules"][-1]
    if last_rule["state"] == RuleStates.PLANED:
        return "rule_creator"

    raise ValueError("Only planed rules can be routed")


def route_from_rule_creator(
    state: State,
) -> Literal["supervisor", "rule_hard_validator"]:
    # if no rules, throw error
    if len(state["rules"]) == 0:
        raise ValueError("No rules to route")

    last_rule = state["rules"][-1]
    if last_rule["state"] == RuleStates.CREATED:
        return "rule_hard_validator"
    elif last_rule["state"] == RuleStates.EXHAUSTED:
        return "supervisor"

    raise ValueError("Only created rules can be routed")


def route_from_rule_hard_validator(
    state: State,
) -> Literal["rule_creator", "rule_soft_validator"]:
    # if no rules, throw error
    if len(state["rules"]) == 0:
        raise ValueError("No rules to route")

    last_rule = state["rules"][-1]
    if last_rule["state"] == RuleStates.HARD_VALIDATED:
        return "rule_soft_validator"
    elif last_rule["state"] == RuleStates.REJECTED:
        return "rule_creator"

    raise ValueError("Only hard validated rules can be routed")


def route_from_rule_soft_validator(
    state: State,
) -> Literal["supervisor", "rule_creator"]:
    # if no rules, throw error
    if len(state["rules"]) == 0:
        raise ValueError("No rules to route")

    last_rule = state["rules"][-1]
    if last_rule["state"] in {RuleStates.ACCEPTED, RuleStates.EXHAUSTED}:
        return "supervisor"
    elif last_rule["state"] == RuleStates.REJECTED:
        return "rule_creator"

    raise ValueError("Only accepted or rejected rules can be routed")


def build_auto_trimmer():
    builder = StateGraph(State)

    # adding nodes
    builder.add_node("supervisor", supervisor)
    builder.add_node("rule_creator", rule_creator)
    builder.add_node("rule_hard_validator", rule_hard_validator)
    builder.add_node("rule_soft_validator", rule_soft_validator)

    # config entry point
    builder.set_entry_point("supervisor")
    builder.set_finish_point("supervisor")

    # adding edges
    builder.add_conditional_edges("supervisor", route_from_supervisor)
    builder.add_conditional_edges("rule_creator", route_from_rule_creator)
    builder.add_conditional_edges("rule_hard_validator", route_from_rule_hard_validator)
    builder.add_conditional_edges("rule_soft_validator", route_from_rule_soft_validator)

    return builder.compile()
