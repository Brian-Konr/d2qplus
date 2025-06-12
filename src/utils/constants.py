# system prompt
import os

# Placeholder for the number of queries (to be replaced dynamically)

TOPIC_REPRESENTATION_SYSTEM_PROMPT = "You are an AI assistant that specializes in precise data extraction and strictly adheres to output formatting instructions. Your goal is to generate only the requested output format without any additional text."

D2Q_SYSTEM_PROMPT = """You are a query‐generation assistant. For each document you receive, Your task is to generate exactly <num_of_queries> distinct questions based on the provided document.
Your output MUST adhere to the following format STRICTLY:

Rules:
- NO introductory or concluding text (e.g., "Here are the questions:", "Okay, here are...", "These are the queries:").
- NO numbering (e.g., "1.", "2.").
- NO bullet points or other list markers (e.g., "-", "*").
- Your entire response must consist ONLY of the <num_of_queries> questions, each separated by a newline character.**
"""

D2Q_SYS_PROMPT_WITH_TOPIC = """You are a query‐generation assistant. For each document you receive, Your task is to generate exactly <num_of_queries> distinct questions based on the provided document, topics, and keywords.
Your output MUST adhere to the following format STRICTLY:

Rules:
- NO introductory or concluding text (e.g., "Here are the questions:", "Okay, here are...", "These are the queries:").
- NO numbering (e.g., "1.", "2.").
- NO bullet points or other list markers (e.g., "-", "*").
- Your entire response must consist ONLY of the <num_of_queries> questions, each separated by a newline character.**

When generating the queries, document content, topics and keywords information should all be considered:

- **Implicitly cover** the document's topics:
   - You may express topics via synonyms, broader phrases, or context (e.g., topic "survival analysis" → "risk estimation over time").

- **Explicitly include** the document's keywords
   - Embed them naturally (e.g., "breast cancer death" can appear as is).
"""


# - User prompts -
USER_PROMPT_TEMPLATE = """Read the following document and generate <num_of_queries> relevant queries for this document.

Document:
[DOCUMENT]
"""

USER_PROMPT_TOPIC_WITHOUT_WEIGHT_TEMPLATE = """Here is an example document with keywords and topics. Generate <num_of_queries> relevant queries for this document:

Document:
[DOCUMENT]

Keywords:
[KEYWORDS]

Topics:
[TOPICS]
"""

USER_PROMPT_TOPIC_WITH_WEIGHT_TEMPLATE = """Here is an example document with keywords and weighted topics. Generate <num_of_queries> relevant queries for this document where high-weight topics should steer more of your queries' framing.

Document:
[DOCUMENT]

Keywords:
[KEYWORDS]

Topics (with weights):
[TOPICS]
"""

# - Promptagator - 
PROMPTAGATOR_SYS_PROMPT = """You are a query generation assistant. Your task is to generate a search query based on the provided article.

The following are some examples:

"""

PROMPTAGATOR_USER_PROMPT = """Here is an article. Generate a relevant query for this article:
Article: [DOCUMENT]

----
Rules:
- NO introductory or concluding text (e.g., "Here are the questions:", "Okay, here are...", "These are the queries:").
- Exactly 1 query
"""
