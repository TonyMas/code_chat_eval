# Evaluation Framework for Chat-Based Code Assistant LLMs

Natural language chat is one of the primary use cases for code assistants like GitHub Copilot or Amazon Q Developer. Users ask questions, the assistant provides answers and code snippets, and the conversation continues with clarifications and additional questions.

Unfortunately, there hasn't been a valid public evaluation system for such chat-based scenarios—until now.

This repository aims to address that gap.

## Key Challenges in Code Assistant Chat Evaluations

1. **Multi-turn Conversations**: Chat-based interactions involve multi-turn dialogues, meaning the evaluation framework must support this. Since the assistant's first response can steer the conversation in an unexpected direction, we can't rely on pre-defined dialogues. Every step of the conversation needs to be evaluated dynamically.
   
2. **Varied User Intentions**: Users might interact with code assistants with different goals—sometimes seeking functional code, other times requesting explanations, or both. The context can also vary significantly, from questions on a repository with numerous unit tests to general inquiries. Thus, the evaluation can't be solely based on unit tests or code execution; it requires a holistic review of the model's responses from multiple angles.

## Current Evaluation Framework

The framework uses different LLMs to evaluate each step of a conversation until the evaluator deems that the code assistant has correctly and completely answered the original question, or until a predefined number of steps have been exhausted.

The process begins by asking the code assistant a question. The response is then evaluated by another LLM based on several criteria. If the evaluator is satisfied with the response, the conversation ends. If not, the evaluator generates a follow-up question, which is passed to the code assistant. This process is repeated until the evaluation is complete.

Framework also allows for the use of multiple evaluators and combine their feedback.

![plot](images/framework.svg)

### Pros and Cons

- **Pros**:
  - The evaluation requires only a set of predefined questions and one or more evaluator models. No need for generating gold-standard answers.
  - The framework tests true multi-turn chat scenarios, where intermediate answers are generated by the evaluated model rather than being drawn from a predefined dataset.
  - The framework is easily extensible to different code assistant scenarios, whether for general queries, repository-specific questions, or code explanations.

- **Cons**:
  - Each use case demands careful selection of evaluator models and extensive prompt engineering. Much of the work for this proof of concept (POC) focused on prompt engineering and defining robust evaluation criteria.
  - The current framework does not directly evaluate the correctness of source code generated by the model by unit tests or code execution. This could be added for specific use cases, but would require additional considerations.

## Implementation Details

### Data

For simplicity, this proof of concept focuses on scenarios where users ask the code assistant general knowledge coding questions that don't require access to specific repositories.

To generate such questions, we use the publicly available StackOverflow dataset:
[StackOverflow Dataset on Kaggle](https://www.kaggle.com/datasets/stackoverflow/stackoverflow)

From this dataset, we extract 200 unanswered Python questions, which serve as the initial questions for our evaluations.

### Evaluators

The evaluators are carefully prompted LLMs that assess the code assistant's answers based on three criteria:
- **Understanding**: Did the code assistant correctly understand the question and provide a relevant solution?
- **Correctness**: Is the solution provided by the assistant accurate and error-free?
- **Completeness**: Is the solution comprehensive, or is additional information required?

If needed, the evaluator also generates a follow-up question for the code assistant.

Experiments have shown that, for the current scenario (answering StackOverflow questions), the best approach is to prompt the evaluator LLM to adopt the persona of a senior engineering mentor guiding an intern.

All evaluator prompts can be found in the [prompt_files](src/mt_chat_code_eval/prompt_files) directory.

### Evaluating the Evaluators

A crucial aspect of this framework is ensuring that the evaluator models are trustworthy. To test this, we created a validation mechanism to assess how well different models perform the evaluation task.

We extracted a separate test set from the StackOverflow dataset, containing both answered and unanswered questions. Positive answers were accepted answers with high positive user scores, while negative answers were those with low negative scores.

The assumption is that the evaluator should mark a positive answer as complete and a negative answer as unsatisfactory by one or more criteria.

|                   | Understanding | Correctness | Completeness | All criteria | Understanding | Correctness | Completeness | All criteria |
| ----------------- | ------------- | ----------- | ------------ | ------------ | ------------- | ----------- | ------------ | ------------ |
| **gpt-4o-2024-08-06** | 96.00%        | 87.00%      | 70.00%       | **69.00%**   | 44.00%        | 26.00%      | 9.00%        | **9.00%**    |
| gpt-4o-2024-05-13 | 100.00%       | 93.00%      | 56.00%       | 56.00%       | 52.00%        | 29.00%      | 4.00%        | 4.00%        |
| gemini-1.5-pro    | 95.00%        | 80.00%      | 48.00%       | 47.00%       | 41.00%        | 25.00%      | 3.00%        | 3.00%        |
| gpt-4o-mini       | 76.00%        | 48.00%      | 5.00%        | 2.00%        | 19.00%        | 8.00%       | 0.00%        | 0.00%        |
| gemini-1.5-flash  | 95.00%        | 70.00%      | 34.00%       | 34.00%       | 48.00%        | 18.00%      | 4.00%        | 4.00%        |

In this proof-of-concept, **gpt-4o-2024-08-06** was chosen as the primary evaluator due to its balanced performance. However, the framework allows for using multiple evaluators, and selecting the best combination is a future task.

Validation results can be found in the [validation_results](validation_results) folder.

### Example Conversations

Several example conversations are included in the [examples](examples) directory:

- [Misunderstood Question](examples/misunderstood-question.txt): A conversation where the model misunderstood the question and failed to provide a correct answer within five steps.
- [Misunderstood Question Assessment](examples/misunderstood-evaluation.txt): Evaluator's assessment of the above conversation.
- [Incomplete Answer](examples/incomplete-answer.txt): A conversation where the model provided an incomplete answer.
- [Incomplete Answer Assessment](examples/incomplete-evaluation.txt): Evaluator's assessment of the above conversation.
- [One-Step Conversation](examples/one-step-conversation.txt): A conversation successfully completed in one step.
- [examples/wrong-language.txt](examples/one-step-conversation.txt): A conversation where code assistant didn't understand the programming language and evaluator corrected it.

## Evaluation results

Due to a limited resources of this POC only handful of models were evaluated, just to show framework capabilities.

|                             | Completed tasks | Average turns in completed tasks |
| --------------------------- | --------------- | ----------------------- |
| gpt-4o-mini                 | 100.00%         | 1.1                     |
| gemini-1.5-flash            | 97.50%          | 1.26                    |
| Llama-3.1-8B-Instruct-Turbo | 81.00%          | 2.05                    |
| CodeLlama-34b-Instruct      | 59.00%          | 2.08                    |

All evaluation results can be found in [evaluation_results](evaluation_results) folder.

## Future Work and Improvements

### Better Evaluation Data

1. The current dataset is from 2022, so the questions are outdated and likely included in the training data of modern LLMs.
2. StackOverflow questions are often too simple for advanced LLMs, so advanced models generally perform well and near 100% quality.

To mitigate these issues:
1. Use the more up-to-date StackOverflow dataset available at [Archive.org](https://archive.org/download/stackexchange), which is regularly updated.
2. Implement a more thorough validation process for selecting harder questions for the evaluation set.

### Improved Evaluator Models

More careful selection and validation of evaluators are needed. Prompt engineering for evaluators could also be significantly improved. 

### Use Multi-Agent Systems as Evaluators

In ideal scenario evaluators should not be individual LLMs that do all evaluation work in one go, but rather multi-agent systems, where agents can do separate tasks - some evaluate code for correctness, some evaluate answer for completeness, some generate follow-up questions.

### Extending the Framework

The framework can easily be extended to evaluate different use cases, such as assessing code assistants on open-source project issues. Additionally, integrating code execution to verify that the generated code works could add further robustness to the evaluation process.

## Usage

To use the framework, install it as a local Python package, then run the evaluation script:

```zsh
pip install -e .

python -m mt_chat_code_eval.run_evaluation --model gpt-4o-mini --evaluators gpt-4o-2024-08-06 --max_steps 5
```

The results will be saved in the [evaluation_results](evaluation_results)  folder.

Model and evaluator names should be valid keys defined in [src/mt_chat_code_eval/llm_fabric.py](src/mt_chat_code_eval/llm_fabric.py) file.

Ensure the following environment variables are set:
```Python
OPENAI_API_KEY # Your OpenAI API key
OPENAI_API_ORG # Your OpenAI organization ID
OPENAI_API_PROJECT # Your OpenAI project ID
GEMINI_API_KEY # Your Gemini AI account key
AIMLAPI_KEY # Your AIMLAPI account key (if you want to use models hosted on aimlapi.com)
```
The project uses dotenv, so you can set these variables in a .env file.