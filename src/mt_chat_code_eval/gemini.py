# This file contains implementation for OpenAI models
import os
from typing import Dict, List

import google.generativeai as gemini

from mt_chat_code_eval.llm_abstract import LLM


# Base class for LLM inference
class Gemini(LLM):

    def __init__(self, model_name: str):
        super().__init__(model_name)

        gemini.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = gemini.GenerativeModel(self.model_name)

        self.conversation: List[Dict[str, str]] = []

    def start_conversation(self, prompt: str) -> str:
        self.chat = self.model.start_chat()
        self.conversation = []

        answer = self.continue_conversation(prompt)

        return answer

    def continue_conversation(self, prompt: str) -> str:
        # Gemini keeps track of the conversation history by itself
        # we just need to send the next prompt and get the response
        try:
            response = self.chat.send_message(
                prompt,
                generation_config=gemini.types.GenerationConfig(
                    temperature=0.3,
                ),
            )
            answer = response.text
        except Exception as e:
            print(e)
            answer = "I cannot answer to this prompt."

        self.conversation.append({"role": "user", "parts": prompt})
        self.conversation.append({"role": "model", "parts": answer})

        return answer

    def get_current_conversation(self) -> List[str]:
        return [message["parts"] for message in self.conversation]
