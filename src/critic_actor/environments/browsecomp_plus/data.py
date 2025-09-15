"""
Data helper functions for BrowseComp Plus Environment
"""
from typing import Any

import tiktoken
from transformers import AutoTokenizer


def process_sample(
    sample: dict[str, Any],
    max_distractors: int | None = None,
    ambiguous_titles: bool = False,
    max_doc_tokens: int | None = None,
    tokenizer: AutoTokenizer | None = None,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    """
    Process sample from BrowseComp Plus into our format
    """
    # sample.keys() is
    # ['query_id', 'query', 'answer', 'gold_docs', 'negative_docs', 'evidence_docs']

    if max_doc_tokens is not None:
        assert tokenizer is not None, "Tokenizer must be provided if max_doc_tokens is provided"

    def get_title(sample_text: str) -> str | None:
        """Get the title from the sample text."""
        title_splits = sample_text.split("---\ntitle: ")[-1].split("\n")
        if len(title_splits) > 1:
            return title_splits[0]
        return None
    
    # Get list of all documents
    all_docs = []
    num_distractors = 0
    num_docs = 0
    max_distractors = max_distractors or len(sample["negative_docs"])
    for k, v in sample.items():
        if k in ["gold_docs", "negative_docs"]:
            for _, doc in enumerate(v):
                if (
                    k == "gold_docs"
                    or ("---\ntitle: " in doc["text"] and num_distractors < max_distractors)
                ):
                    doc_title = get_title(doc["text"])
                    if doc_title is not None:  # the usual case
                        doc_text = doc["text"][len("---\ntitle: " + doc_title):]
                    else:
                        doc_title = doc["text"].split("\n")[0][:32] + "..."
                        doc_text = doc["text"][len(doc_title) - len("..."):]
                    if max_doc_tokens is not None:
                        model_or_encoding = "cl100k_base"  # approx with tiktoken
                        enc = tiktoken.get_encoding(model_or_encoding)
                        doc_text = enc.decode(enc.encode(doc_text)[:max_doc_tokens])
                        # doc_text = tokenizer.decode(tokenizer.encode(doc_text)[:max_doc_tokens])
                        # doc_text = doc_text[:max_doc_tokens]
                    doc_text = doc_text.strip()
                    all_docs.append({
                        "title": (
                            doc_title.strip() if not ambiguous_titles else
                            f"title {num_docs:>03d}."
                        ),
                        "url": doc["url"],
                        "text": doc_text,
                        "is_gold": k == "gold_docs",
                        "category": k,
                    })
                    if k == "negative_docs":
                        num_distractors += 1
                    num_docs += 1

    # Create lookup dictionary for all documents by title
    all_docs_dict = {
        _doc["title"]: {
            k: v for k, v in _doc.items() if k != "title"
        }
        for _doc in all_docs
    }
    all_titles = sorted(list(all_docs_dict.keys()))
    
    return all_docs_dict, all_titles, all_docs


def get_search_tool_desc(titles: list[str]) -> dict:
    """Get the description of the search tool"""
    return {
        "type": "function",
        "name": "search",
        "description": "Search the web for information about the given titles.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": (
                        "The title to search for. Can only be one of:\n- "
                        + "\n- ".join([f"'{t}'" for t in titles])
                    ),
                },
            },
            "required": ["title"],
        },
    }


def render_prompt(
    query: str,
    all_titles: list[str],
    include_titles_in_prompt: bool = False,
) -> tuple[str, dict[str, Any]]:
    """
    Render prompt for BrowseComp Plus

    Returns:
    - prompt (str): The prompt to be used for the model
    - search_tool_desc (dict[str, Any]): The description of the search tool
    """
    initial_msg = (
        f"## Instruction\nGiven a list of website titles, think and search to answer the following question:\n'''\n{query}\n'''\n\n"
        "- Thoughts can reason about the current situation.\n"
        "- The `search` tool can be used to look up information based on the titles. "
        f"Only search the titles provided. Only call the `search` tool once per turn.\n\n"
        "Your final answer should be a concise sentence, in the following format: "
        "'Final Answer: <put your answer here>'."
        # "\n\nIt's critical your answer is concise and following the format strictly."
    )
    final_msg = (
        f"\n\n## Instruction (again)\nNow answer the original question. Recall the question is:\n'''\n{query}\n'''\n\n"
        "VERY IMPORTANT: You may only use the provided `search` tool once per turn, and only use the given titles to answer"
        " the question. If you provide a title not in the given titles, the search will fail."
    )
    search_tool_desc = get_search_tool_desc(all_titles)

    if include_titles_in_prompt:
        tool_msg = (
            "\n\n## Tool Calling\n"
            "You can only search the following titles:\n"
            "\n- " + "\n- ".join([f"'{t}'" for t in all_titles])
        )
    else:
        tool_msg = ""
    prompt = initial_msg + tool_msg + final_msg
    return prompt, search_tool_desc
