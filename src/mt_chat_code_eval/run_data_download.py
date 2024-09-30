import argparse
import os

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google.cloud import bigquery

# Load local environment variables
load_dotenv()

# Create a BigQuery client
# This will use the credentials stored in the GOOGLE_APPLICATION_CREDENTIALS
# Plus project name from the environment variable GOOGLE_PROJECT_NAME
big_query_client = bigquery.Client(os.getenv("GOOGLE_PROJECT_NAME"))

# Query to get the data from the Stackoverflow dataset for evaluation
# For the evaluation data, we will get the most recent questions that
# do not have an accepted answer
eval_data_query = """
SELECT
CONCAT('https://stackoverflow.com/questions/', CAST(id as STRING)) as url,
id,
body,
creation_date
FROM `bigquery-public-data.stackoverflow.posts_questions`
WHERE
tags LIKE "%python%" AND
accepted_answer_id IS NULL
ORDER BY creation_date DESC
LIMIT {count}
"""

# Query to get the data from the Stackoverflow dataset to validate automatic evaluators
# For the validation data, we will get the most recent questions that have both accepted
# answer and at least one non-accepted answer
# To strengthen the signal that we get from the data,
# we will get accepted answers with a high positive score
# and non-accepted answers with a low negative score
# This way we will have both positive and negative examples for evaluators to validate
validation_data_query = """
SELECT
url,
id,
body,
creation_date,
accepted_answer_id,
answer_id,
answer_body,
answer_score
FROM
(
    SELECT
    CONCAT('https://stackoverflow.com/questions/', CAST(id as STRING)) as url,
    id,
    body,
    creation_date,
    accepted_answer_id,
    FROM `bigquery-public-data.stackoverflow.posts_questions`
    WHERE
    tags LIKE "%python%" AND
    accepted_answer_id IS NOT NULL AND
    answer_count > 1
) as questions
LEFT JOIN
(
    SELECT
    id as answer_id,
    body as answer_body,
    parent_id as question_id,
    score as answer_score
    FROM `bigquery-public-data.stackoverflow.posts_answers`
) ON id = question_id
WHERE
answer_id IS NOT NULL AND
(
{answer_condition}
)
ORDER BY creation_date DESC
LIMIT {count}
"""

# Query to get the positive examples for the validation data
validation_pos_query = """
( ( answer_score > 3 ) AND ( accepted_answer_id = answer_id ) )
"""

# Query to get the negative examples for the validation data
validation_neg_query = """
( ( answer_score < -2 ) AND ( accepted_answer_id != answer_id ) )
"""


def get_data(query: str) -> pd.DataFrame:
    query_job = big_query_client.query(query)
    return query_job.result().to_dataframe()


def remove_html_tags(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--evaluation_count", type=int, default=200)
    parser.add_argument("--validation_count", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="data")

    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Getting evaluation data
    eval_query = eval_data_query.format(count=args.evaluation_count)
    eval_data = get_data(eval_query)
    eval_data["question"] = eval_data["body"].apply(remove_html_tags)

    eval_data = eval_data.reset_index(drop=True)

    eval_data.to_parquet(os.path.join(args.output_dir, "evaluation_data.parquet"))

    # Getting validation data

    # For validation data, we need to get positive and negative examples separately
    valid_pos_query = validation_data_query.format(
        answer_condition=validation_pos_query, count=int(args.validation_count) // 2
    )
    valid_neg_query = validation_data_query.format(
        answer_condition=validation_neg_query, count=int(args.validation_count) // 2
    )

    valid_pos_data = get_data(valid_pos_query)
    valid_neg_data = get_data(valid_neg_query)

    valid_data = pd.concat([valid_pos_data, valid_neg_data], ignore_index=True)

    valid_data["is_accepted"] = (
        valid_data["accepted_answer_id"] == valid_data["answer_id"]
    )

    valid_data["question"] = valid_data["body"].apply(remove_html_tags)
    valid_data["answer"] = valid_data["answer_body"].apply(remove_html_tags)

    valid_data = valid_data.reset_index(drop=True)

    valid_data.to_parquet(os.path.join(args.output_dir, "validation_data.parquet"))
