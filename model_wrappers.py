import math
import os
from typing import Dict, List

import anthropic
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from data import Article, Summary, load_articles
from prompts import (
    INDIVIDUAL_QUALITY_PROMPT,
    INDIVIDUAL_QUALITY_SYSTEM_PROMPT,
    INDIVIDUAL_RECOGNITION_PROMPT,
    INDIVIDUAL_RECOGNITION_SYSTEM_PROMPT,
    PAIRWISE_PREFERENCE_PROMPT,
    PAIRWISE_PREFERENCE_SYSTEM_PROMPT,
    PAIRWISE_RECOGNITION_PROMPT,
    PAIRWISE_RECOGNITION_SYSTEM_PROMPT,
    SUMMARIZATION_PROMPTS,
    SUMMARIZATION_SYSTEM_PROMPTS,
)

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_LENGTH = 1024
ARTICLES = load_articles()


class ModelWrapper:
    """Wrapper for LLM evaluators"""

    def __init__(self, model_id: str, model_name):
        """
        Initializes the LLM debater with the specified model.

        Args:
            model_id (str): A short identifier for the model ("llama2_7b")
            model_name (str): The name of the model to load from HF/API
        """
        self.model_id = model_id
        self.model_name = model_name

    def _response(self, system_prompt: str, user_prompt: str, words_in_mouth="") -> str:
        """Generates model output"""
        raise NotImplementedError

    def _get_token_probs(
        self,
        response_tokens: List[str],
        system_prompt: str,
        user_prompt: str,
        words_in_mouth="",
    ) -> Dict[str, float]:
        """Generates token probabilities"""
        raise NotImplementedError

    def summarize(self, article: Article) -> Summary:
        """Generates a summary for the given article."""
        system_prompt = SUMMARIZATION_SYSTEM_PROMPTS[article.dataset]
        prompt = SUMMARIZATION_PROMPTS[article.dataset].format(article=article.text)

        return self._response(system_prompt, prompt)

    def pairwise_recognition_score(self, summary1: Summary, summary2: Summary) -> float:
        """Returns the model's confidence that the first summary is its own output."""
        assert summary1.article_id == summary2.article_id

        system_prompt = PAIRWISE_RECOGNITION_SYSTEM_PROMPT
        prompt = PAIRWISE_RECOGNITION_PROMPT.format(
            article=ARTICLES[summary1.article_id].text,
            summary1=summary1.text,
            summary2=summary2.text,
        )
        return self._get_token_probs(["1", "2"], system_prompt, prompt)

    def pairwise_preference_score(self, summary1: Summary, summary2: Summary) -> float:
        """Returns the model's confidence that the first summary is better than the second."""
        assert summary1.article_id == summary2.article_id

        system_prompt = PAIRWISE_PREFERENCE_SYSTEM_PROMPT
        prompt = PAIRWISE_PREFERENCE_PROMPT.format(
            article=ARTICLES[summary1.article_id].text,
            summary1=summary1.text,
            summary2=summary2.text,
        )
        return self._get_token_probs(["1", "2"], system_prompt, prompt)

    def individual_recognition_score(self, summary: Summary) -> float:
        """Returns the model's confidence that the summary is its own output."""
        system_prompt = INDIVIDUAL_RECOGNITION_SYSTEM_PROMPT
        prompt = INDIVIDUAL_RECOGNITION_PROMPT.format(
            article=ARTICLES[summary.article_id].text, summary=summary.text
        )
        return self._get_token_probs(["Yes", "No"], system_prompt, prompt)

    def individual_quality_score(self, summary: Summary) -> float:
        """Returns the model's weighted Likert-score rating (1-5) of the summary."""
        system_prompt = INDIVIDUAL_QUALITY_SYSTEM_PROMPT
        prompt = INDIVIDUAL_QUALITY_PROMPT.format(
            article=ARTICLES[summary.article_id].text, summary=summary.text
        )
        scores = self._get_token_probs(["1", "2", "3", "4", "5"], system_prompt, prompt)
        return sum(int(k) * v for k, v in scores.items())


class HuggingFaceWrapper(ModelWrapper):
    def __init__(self, model_id: str, model_name: str):
        super().__init__(model_id, model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            token=HF_TOKEN,
            # torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _format_prompt(
        self, system_prompt: str, user_prompt: str, words_in_mouth=""
    ) -> str:
        """Formats the prompt for the model"""
        raise NotImplementedError

    def _response(self, system_prompt: str, user_prompt: str, words_in_mouth="") -> str:
        """Generates model output"""
        formatted_prompt = self._format_prompt(
            system_prompt, user_prompt, words_in_mouth
        )
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(
            self.model.device
        )
        output = self.model.generate(
            input_ids,
            max_new_tokens=MAX_LENGTH,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,  # Set pad_token_id to EOS token ID to avoid padding
        )
        decoded = self.tokenizer.decode(
            output[0][input_ids.shape[1] :], skip_special_tokens=True
        )  # Decode only the generated tokens
        return decoded

    def _get_token_probs(
        self,
        response_tokens: List[str],
        system_prompt: str,
        user_prompt: str,
        words_in_mouth="",
    ) -> Dict[str, float]:
        """Generates token probabilities"""
        formatted_prompt = self._format_prompt(
            system_prompt, user_prompt, words_in_mouth
        )
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(
            self.model.device
        )
        output = self.model(input_ids).logits[0, -1, :]
        probs = output.softmax(dim=0)

        correct_answer_prob = probs[
            self.tokenizer.encode(response_tokens[0])[-1]
        ].item()
        incorrect_answer_prob = probs[
            self.tokenizer.encode(response_tokens[1])[-1]
        ].item()
        return correct_answer_prob / (correct_answer_prob + incorrect_answer_prob)


# meta-llama/Llama-2-7b-chat-hf, etc
class Llama2Wrapper(HuggingFaceWrapper):
    def _format_prompt(
        self, system_prompt: str, user_prompt: str, words_in_mouth=""
    ) -> str:
        """Formats the prompt for the model"""
        return f"""<s>[INST] <<SYS>>
        {system_prompt}
        <</SYS>>
        {user_prompt} [/INST] {words_in_mouth}""".strip()

    def _response(self, system_prompt: str, user_prompt: str, words_in_mouth="") -> str:
        return super()._response(system_prompt, user_prompt, words_in_mouth if words_in_mouth else "Here")

# meta-llama/Meta-Llama-3-8B-Instruct, etc
class Llama3Wrapper(HuggingFaceWrapper):
    def _format_prompt(
        self, system_prompt: str, user_prompt: str, words_in_mouth=""
    ) -> str:
        """Formats the prompt for the model"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{words_in_mouth}"""

    def _response(self, system_prompt: str, user_prompt: str, words_in_mouth="") -> str:
        return super()._response(system_prompt, user_prompt, words_in_mouth if words_in_mouth else "Here")


# google/gemma-2-9b, google/gemma-2-27b
class Gemma2Wrapper(HuggingFaceWrapper):
    def _format_prompt(
        self, system_prompt: str, user_prompt: str, words_in_mouth=""
    ) -> str:
        """Formats the prompt for the model"""
        return f"""<start_of_turn>user\n{system_prompt}\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n{words_in_mouth}"""


# gpt-4o-2024-05-13, gpt-4-turbo-2024-04-09, gpt-4-0613, gpt-3.5-turbo-0125
class GPTWrapper(ModelWrapper):
    def __init__(self, model_id: str, model_name: str):
        super().__init__(model_id, model_name)
        self.client = OpenAI()

    def _response(self, system_prompt: str, user_prompt: str, words_in_mouth="") -> str:
        """Generates model output using OpenAI's API"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=MAX_LENGTH,
            n=1,
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    def _get_token_probs(
        self,
        response_tokens: List[str],
        system_prompt: str,
        user_prompt: str,
        words_in_mouth="",
    ) -> Dict[str, float]:
        """Generates token probabilities using OpenAI's API"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=MAX_LENGTH,
            n=1,
            temperature=0,
        )

        token_probs = {token: 0 for token in response_tokens}
        logprobs = response.choices[0].logprobs.top_logprobs[0]

        for token, logprob in logprobs.items():
            if token in token_probs:
                token_probs[token] = math.exp(logprob)

        total_prob = sum(token_probs.values())
        return {k: v / total_prob for k, v in token_probs.items()}


# claude-3-5-sonnet-20240620, claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
class ClaudeWrapper(ModelWrapper):
    def __init__(self, model_id: str, model_name: str):
        super().__init__(model_id, model_name)
        self.client = anthropic.Anthropic()

    def _response(self, system_prompt: str, user_prompt: str, words_in_mouth="") -> str:
        """Generates model output using Anthropic's API"""
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=MAX_LENGTH,
            temperature=0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}],
                }
            ],
        )

        return message.content[0].text

    def _get_token_probs(
        self,
        response_tokens: List[str],
        system_prompt: str,
        user_prompt: str,
        words_in_mouth="",
    ) -> Dict[str, float]:
        """Generates token probabilities using Anthropic's API"""
        raise NotImplementedError("Anthropic does not provide token probabilities")
