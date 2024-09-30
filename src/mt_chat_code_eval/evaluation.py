# Description: This module contains the functions
# to evaluate a conversation using the LLM evaluator.

import re
from typing import List, Tuple, Union

from mt_chat_code_eval.llm_abstract import LLM
from mt_chat_code_eval.prompts import (
    evaluation_end_prompt,
    evaluation_qa_prompt,
    evaluation_start_prompt,
)


def _build_conversation_prompt(conversation: list) -> str:
    qa_pairs = [(q, a) for q, a in zip(conversation[::2], conversation[1::2])]

    prompt = evaluation_start_prompt

    for q, a in qa_pairs:
        prompt += evaluation_qa_prompt.format(question=q, answer=a)

    prompt += evaluation_end_prompt

    return prompt


def _concert_to_bool(string: str) -> Union[bool, None]:
    if string.casefold() == "yes":
        return True
    elif string.casefold() == "no":
        return False
    else:
        return None


def _section_to_regex(section: str) -> str:
    return section.replace(" ", "\\s*") + "((?:.|\\n)*)"


def evaluate_conversation(
    llm_evaluator: LLM, conversation: List[str], retries: int = 1
) -> Tuple[Union[str, None], Union[bool, None], Union[bool, None], Union[bool, None]]:
    # We gather all conversation history in one big prompt for evaluation
    prompt = _build_conversation_prompt(conversation)

    response = llm_evaluator.start_conversation(prompt)

    response_sections = [
        "### Follow-up question",
        "### Understanding",
        "### Correctness",
        "### Completeness",
    ]

    regex = "".join([_section_to_regex(section) for section in response_sections])

    response_regex = re.compile(regex, flags=re.IGNORECASE)

    response_match = response_regex.search(response)

    if response_match:
        followup = response_match.group(1).strip()
        understanding = _concert_to_bool(response_match.group(2).strip())
        correctness = _concert_to_bool(response_match.group(3).strip())
        completeness = _concert_to_bool(response_match.group(4).strip())
    else:
        followup = None
        understanding = None
        correctness = None
        completeness = None

    if (
        understanding is None
        or correctness is None
        or completeness is None
        or (completeness is False and followup is None)
    ):
        if retries > 0:
            # Sometimes the model may not provide all the required information
            # We retry the evaluation in this case
            return evaluate_conversation(llm_evaluator, conversation, retries - 1)

    return followup, understanding, correctness, completeness
