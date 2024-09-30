# This module contains the functions to run a
# conversation with a model and a LLM evaluator.

from typing import List, Tuple

import pandas as pd

from mt_chat_code_eval.evaluation import evaluate_conversation
from mt_chat_code_eval.llm_abstract import LLM


def build_conversation(
    prompt: str, model: LLM, evaluators: List[LLM], max_steps: int = 5
) -> Tuple[List[str], bool, pd.DataFrame]:
    model.start_conversation(prompt)

    evaluations_df = pd.DataFrame(
        columns=[
            "model_name",
            "step",
            "followup",
            "understanding",
            "correctness",
            "completeness",
        ]
    )

    for i in range(max_steps):
        for evaluator in evaluators:
            followup, understanding, correctness, completeness = evaluate_conversation(
                evaluator, model.get_current_conversation()
            )

            evaluations_df = pd.concat(
                [
                    evaluations_df,
                    pd.DataFrame(
                        [
                            {
                                "model_name": evaluator.model_name,
                                "step": i,
                                "followup": followup,
                                "understanding": understanding,
                                "correctness": correctness,
                                "completeness": completeness,
                            }
                        ]
                    ),
                ]
            )

        current_eval = evaluations_df[evaluations_df["step"] == i]

        # We are checking following conditions to stop the conversation:
        # - All evaluators agree that the model understands the questions
        # - All evaluators agree that the model provides correct answers
        # - At least one evaluator agrees that the model provides full answers
        should_stop = (
            current_eval["understanding"].all(skipna=True)
            & current_eval["correctness"].all(skipna=True)
            & current_eval["completeness"].any(skipna=True)
        )

        if should_stop:
            break
        else:
            follow_up_df = current_eval[
                ~current_eval["completeness"]
                & ~current_eval["followup"].isna()
                & ~current_eval["followup"].str.strip().eq("")
            ]

            if len(follow_up_df) > 0:
                model.continue_conversation(follow_up_df.sample(1).iloc[0]["followup"])
            else:
                # It should be a rare case when conversation is not complete
                # but no follow-up questions are found
                # We will consider in this case that the model failed the conversation
                break

    # We will consider conversation successful if it was stopped
    # by evaluators and not because we reached the max steps
    is_successful = should_stop

    return model.get_current_conversation(), is_successful, evaluations_df
