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

# # few shot query set 
D2Q_FEW_SHOT_SYS_PROMPT_WITH_TOPIC = """You are a query‐generation assistant. For each document you receive, Your task is to generate exactly <num_of_queries> distinct questions based on the provided document, topics, and keywords.
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

The following are some examples:

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

PROMPTAGATOR_USER_PROMPT = """Here is an article.
Article: 

[DOCUMENT]

Generate a relevant query for this article:
----
Rules:
- NO introductory or concluding text (e.g., "Here are the questions:", "Okay, here are...", "These are the queries:").
- Exactly 1 query
"""


PROMPTAGATOR_SET_GEN_SYS_PROMPT = """You are a query generation assistant. Your task is to generate <num_of_queries> relevant queries based on the provided article.

The following are some examples:

"""

PROMPTAGATOR_SET_GEN_USER_PROMPT = """Here is an article:
Article:
<document>


Generate <num_of_queries> relevant queries for this article based on the following keywords:
<keywords>

----
Rules:
- NO introductory or concluding text (e.g., "Here are the questions:", "Okay, here are...", "These are the queries:").
- Exactly <num_of_queries> queries, each separated by a newline character.
"""


PROMPTAGATOR_SET_GEN_TOPIC_USER_PROMPT = """You are an expert assistant in crafting search queries that cover specified topics and make use of given keywords for the provided article.

Article:
<document>

Topics:
<topics>

Keywords:
<keywords>

Task:
Generate exactly {num_of_queries} search queries for the above article so that:
1. Together they cover each of the listed topics at least once.
2. Each query uses one or more of the provided keywords in a natural way.

----
Rules:
- NO introductory or concluding text (e.g., "Here are the questions:", "Okay, here are...", "These are the queries:").
- Exactly <num_of_queries> queries, each separated by a newline character.
- Queries must be relevant to the article.
- Queries should collectively cover all topics and use the keywords provided.
"""


PROMPTAGATOR_SET_GEN_NO_TOPIC_KEYWORDS_USER_PROMPT = """You are an expert assistant in crafting search queries for the provided article:

Article:
<document>

Generate exactly <num_of_queries> search queries for the above article.

----
Rules:
- NO introductory or concluding text (e.g., "Here are the questions:", "Okay, here are...", "These are the queries:").
- Exactly <num_of_queries> queries, each separated by a newline character.
- Queries must be relevant to the article.
"""


LLM_EXTRACT_KEYWORD_USER_PROMPT = """You will receive a document along with a set of candidate keywords. Your task is to select the keywords that best align with the core theme of the document. Exclude keywords that are too broad or less relevant. You may list up to <final_keyword_num> keywords, using only the keywords in the candidate keyword set.

Document:
<document>

Candidate keyword set:
<keywords>

IMPORTANT: Respond with ONLY valid JSON in this exact format:
{"keywords": ["keyword1", "keyword2", "keyword3"]}

Do not include any explanations, markdown formatting, or additional text. Just the raw JSON object."""