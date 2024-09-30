from importlib import resources as impresources

from mt_chat_code_eval import prompt_files

start_file = impresources.files(prompt_files) / "evaluation.start.prompt"
with start_file.open("r", encoding="utf-8") as file:
    evaluation_start_prompt = file.read()

qa_file = impresources.files(prompt_files) / "evaluation.qa.prompt"
with qa_file.open("r", encoding="utf-8") as file:
    evaluation_qa_prompt = file.read()

end_file = impresources.files(prompt_files) / "evaluation.end.prompt"
with end_file.open("r", encoding="utf-8") as file:
    evaluation_end_prompt = file.read()
