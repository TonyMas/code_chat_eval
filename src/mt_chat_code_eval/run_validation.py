import argparse
import datetime
import os
from typing import Dict, Union

import pandas as pd
import tqdm
from dotenv import load_dotenv
from slugify import slugify

from mt_chat_code_eval.evaluation import evaluate_conversation
from mt_chat_code_eval.llm_abstract import LLM
from mt_chat_code_eval.llm_fabric import load_llm

# Load local environment variables
load_dotenv()

tqdm.tqdm.pandas()


def _evaluate_row(model: LLM, row: pd.Series) -> Dict[str, Union[str, bool, int, None]]:
    followup, understanding, correctness, completeness = evaluate_conversation(
        model, [row["question"], row["answer"]]
    )
    return {
        "follow_up": followup,
        "understanding": understanding,
        "correctness": correctness,
        "completeness": completeness,
        "is_accepted": row["is_accepted"],
        "score": row["answer_score"],
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--validation_data", type=str, default="data/validation_data.parquet"
    )
    parser.add_argument("--output_dir", type=str, default="validation_results")

    args, _ = parser.parse_known_args()

    model = load_llm(args.model)

    validation_data = pd.read_parquet(args.validation_data)

    validation_results = validation_data.progress_apply(
        lambda x: _evaluate_row(model, x), axis=1, result_type="expand"
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    date = datetime.date.today().isoformat()

    result_name = f"{args.model}___{date}.parquet"
    result_name = slugify(result_name)

    validation_results.to_parquet(os.path.join(args.output_dir, result_name))
