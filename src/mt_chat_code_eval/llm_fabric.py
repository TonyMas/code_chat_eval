from mt_chat_code_eval.gemini import Gemini
from mt_chat_code_eval.llm_abstract import LLM
from mt_chat_code_eval.open_ai import AIMLAPI, OpenAI

model_list = {
    "gpt-4o": OpenAI,
    "gpt-4o-2024-08-06": OpenAI,
    "gpt-4o-mini": OpenAI,
    "gpt-4-turbo": OpenAI,
    "gpt-3.5-turbo": OpenAI,
    "gemini-1.5-flash": Gemini,
    "gemini-1.5-pro": Gemini,
    "gemini-1.0-pro": Gemini,
    "codellama/CodeLlama-7b-Instruct-hf": AIMLAPI,
    "codellama/CodeLlama-13b-Instruct-hf": AIMLAPI,
    "codellama/CodeLlama-34b-Instruct-hf": AIMLAPI,
    "codellama/CodeLlama-70b-Instruct-hf": AIMLAPI,
    "togethercomputer/CodeLlama-7b-Instruct": AIMLAPI,
    "togethercomputer/CodeLlama-13b-Instruct": AIMLAPI,
    "togethercomputer/CodeLlama-34b-Instruct": AIMLAPI,
    "deepseek-ai/deepseek-coder-33b-instruct": AIMLAPI,
    "WizardLM/WizardCoder-Python-34B-V1.0": AIMLAPI,
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": AIMLAPI,
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": AIMLAPI,
}


def load_llm(model_name: str) -> LLM:
    if model_name not in model_list:
        raise ValueError(f"Model {model_name} not found in the available models")
    return model_list[model_name](model_name)  # type: ignore
