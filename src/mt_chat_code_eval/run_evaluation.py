import argparse
import datetime
import os
from typing import Dict, List, Union

import pandas as pd
import tqdm
from dotenv import load_dotenv
from slugify import slugify

from mt_chat_code_eval.conversation import build_conversation
from mt_chat_code_eval.llm_abstract import LLM
from mt_chat_code_eval.llm_fabric import load_llm

# Load local environment variables
load_dotenv()

tqdm.tqdm.pandas()


def _first_const_true(series: pd.Series) -> Union[int, None]:
    if series.iloc[-1] is True:
        for i in range(-2, -len(series) - 1, -1):
            if series.iloc[i] is False:
                return series.index[i + 1]
        return series.idxmin()
    else:
        return None


def _steps_to_achieve(eval_df: pd.DataFrame, criteria: str) -> int:
    step_index = _first_const_true(eval_df[criteria])
    if step_index is None:
        return eval_df["step"].max() + 1
    else:
        return eval_df.loc[step_index]["step"] + 1


def get_eval_metrics(eval_df: pd.DataFrame) -> Dict[str, int]:
    eval_grouped = (
        eval_df.groupby("step")
        .agg(
            {
                "understanding": "all",
                "correctness": "all",
                "completeness": "any",
            }
        )
        .reset_index()
    )

    return {
        "steps_total": eval_grouped["step"].max() + 1,
        "steps_to_understanding": _steps_to_achieve(eval_grouped, "understanding"),
        "steps_to_correctness": _steps_to_achieve(eval_grouped, "correctness"),
        "steps_to_completeness": _steps_to_achieve(eval_grouped, "completeness"),
    }


def _evaluate_row(
    model: LLM, evaluators: List[LLM], max_steps: int, row: pd.Series
) -> Dict[str, object]:
    conversation, is_successful, evaluations = build_conversation(
        row["question"], model=model, evaluators=evaluators, max_steps=max_steps
    )
    result = {
        "conversation": conversation,
        "complete": is_successful,
    } | get_eval_metrics(evaluations)
    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--evaluators",
        type=str,
        nargs="+",
        default=["gpt-4o-2024-08-06"],
    )
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument(
        "--evaluation_data", type=str, default="data/evaluation_data.parquet"
    )
    parser.add_argument("--output_dir", type=str, default="evaluation_results")

    args, _ = parser.parse_known_args()

    model = load_llm(args.model)

    evaluators = [load_llm(evaluator) for evaluator in args.evaluators]

    evaluation_data = pd.read_parquet(args.evaluation_data)

    evaluation_results = evaluation_data.progress_apply(
        lambda x: _evaluate_row(model, evaluators, args.max_steps, x),
        axis=1,
        result_type="expand",
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    date = datetime.date.today().isoformat()

    result_name = f"{args.model}___{args.max_steps}___vs___"
    result_name += f"{'___and___'.join(args.evaluators)}___{date}.parquet"

    result_name = slugify(result_name)

    evaluation_results.to_parquet(os.path.join(args.output_dir, result_name))
