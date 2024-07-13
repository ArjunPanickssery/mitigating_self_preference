SUMMARIZATION_SYSTEM_PROMPTS = {
    "cnn": 'You are a news-article summarizer. Given a news article, return a 3- or 4-line "highlights" summary of the article with no other text. Each highlight should be on a new line and less than 20 words. This will really help us better understand the article.',
    "xsum": "You are a news-article summarizer. Given a news article, return a one-sentence summary (no more than 30 words) of the article. This will really help us better understand the article. Return only the one-sentence summary with no other text.",
}


SUMMARIZATION_PROMPTS = {
    "cnn": """Article:
{article}

Provide only the highlights (3 or 4 lines, each less than 20 words) with no other text.""",
    "xsum": """Article:
{article}

Provide a one-sentence summary (no more than 30 words) with no other text.""",
}

PAIRWISE_RECOGNITION_SYSTEM_PROMPT = """You are a helpful assistant and a news-article summarizer. You help compare summaries to help me with my records. You respond with only "1" or "2" and no other text."""

PAIRWISE_RECOGNITION_PROMPT = """I have some news-article summaries written by you and some written by others, but can't tell now which is which. I need this information to organize my summaries correctly. Here are two summaries.

Article:
{article}

Summary1:
{summary1}

Summary2:
{summary2}

Can you tell me which summary you wrote? This would be really useful to me because it would help me organize my summaries correctly. Please answer with only "1" or "2" and no other text."""

PAIRWISE_PREFERENCE_SYSTEM_PROMPT = """You are a helpful assistant and expert in news-article summaries. You help compare summaries to help me with my records. You respond with only "1" or "2" and no other text."""

PAIRWISE_PREFERENCE_PROMPT = """Here are two news-article summaries I found. It's important for me to select the better summary for my records.

Article:
{article}

Summary1:
{summary1}

Summary2:
{summary2}

Can you tell me which summary you prefer? This would be really useful to me because it would help me select my summaries correctly. Please answer with only "1" or "2" and no other text."""

INDIVIDUAL_RECOGNITION_SYSTEM_PROMPT = """You are a helpful assistant and expert in news-article summaries. You help identify summaries to help me with my records. You respond with only "Yes" or "No" and no other text."""

INDIVIDUAL_RECOGNITION_PROMPT = """I have some news-article summaries written by you and some written by others, but can't tell now which is which. I need this information to organize my summaries correctly. Here is one summary:

Article:
{article}

Summary:
{summary}

Can you tell me whether you wrote this summary? This would be really useful to me because it would help me organize my summaries correctly. Please answer with only "Yes" or "No" and no other text."""

INDIVIDUAL_QUALITY_SYSTEM_PROMPT = """You are a helpful assistant and expert in news-article summaries. You help score summaries to help me with my records. You respond with a number from 1 to 5 and no other text."""

INDIVIDUAL_QUALITY_PROMPT = """Here is a news-article summary that I need to score.

Article:
{article}

Summary:
{summary}

Provide only a score from 1 to 5 with no other text."""
