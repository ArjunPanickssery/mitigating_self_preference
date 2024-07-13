from dataclasses import asdict, dataclass
from typing import List
import json
import os

from datasets import load_dataset

SOURCES = ["human", "claude", "gpt35", "gpt4", "llama"]
ARTICLES_LOC = "data/articles.json"
SUMMARIES_LOC = "data/summaries.json"


@dataclass
class Article:
    dataset: str
    id: str
    text: str


@dataclass
class Summary:
    dataset: str
    article_id: str
    source: str
    text: str


def save_to_json(dictionary: dict, file_name: str):
    # Create directory if not present
    directory = os.path.dirname(file_name)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_name, "w") as f:
        json.dump(dictionary, f)


def load_from_json(file_name: str) -> dict:
    with open(file_name, "r") as f:
        return json.load(f)


def save_articles(articles: List[Article]):
    with open(ARTICLES_LOC, "w") as f:
        json.dump(
            {id: asdict(article) for id, article in articles.items()},
            f,
            indent=2,
        )


def load_articles():
    with open(ARTICLES_LOC, "r") as f:
        return {
            id: Article(**article_data) for id, article_data in json.load(f).items()
        }


def save_summaries(summaries: List[Summary]):
    with open(SUMMARIES_LOC, "w") as f:
        json.dump([asdict(summary) for summary in summaries], f, indent=2)


def load_summaries():
    with open(SUMMARIES_LOC, "r") as f:
        return [Summary(**summary_data) for summary_data in json.load(f)]
