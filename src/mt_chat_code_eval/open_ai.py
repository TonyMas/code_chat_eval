# This file contains implementation for OpenAI models
import os
from typing import Dict, List

from openai import OpenAI as OpenAI_API

from mt_chat_code_eval.llm_abstract import LLM


# Base class for LLM inference
class OpenAI_Base(LLM):

    def __init__(self, model_name: str, client: OpenAI_API):
        super().__init__(model_name)

        self.client = client

        self.conversation: List[Dict[str, str]] = []

    def _get_response(
        self, conversation: List[Dict[str, str]], retries: int = 1
    ) -> str:

        try:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=conversation, temperature=0.3
            )

            answer = response.choices[0].message.content
        except Exception as e:
            print(e)
            if retries > 0:
                return self._get_response(conversation, retries - 1)
            else:
                answer = "I cannot answer to this prompt."

        return answer

    def start_conversation(self, prompt: str) -> str:
        self.conversation = []
        return self.continue_conversation(prompt)

    def continue_conversation(self, prompt: str) -> str:
        # Open AI cannot store history of the conversation,
        # we need to query it with the full history each turn
        self.conversation.append({"role": "user", "content": prompt})

        answer = self._get_response(self.conversation)

        self.conversation.append({"role": "assistant", "content": answer})

        return answer

    def get_current_conversation(self) -> List[str]:
        return [message["content"] for message in self.conversation]


class OpenAI(OpenAI_Base):
    def __init__(self, model_name: str):
        client = OpenAI_API(
            organization=os.getenv("OPENAI_API_ORG"),
            project=os.getenv("OPENAI_API_PROJECT"),
        )

        super().__init__(model_name, client)


class AIMLAPI(OpenAI_Base):
    def __init__(self, model_name: str):
        client = OpenAI_API(
            api_key=os.getenv("AIMLAPI_KEY"), base_url="https://api.aimlapi.com/v1"
        )

        super().__init__(model_name, client)
