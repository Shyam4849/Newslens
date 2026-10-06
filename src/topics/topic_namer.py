"""Turn keywords into a human-readable topic name.

Default: rule-based (no API). Optional: LLM naming (keywords -> name only),
used when ANTHROPIC_API_KEY is set and the toggle is enabled.
"""
from __future__ import annotations

import os


def rule_based_name(keywords: list[str], n: int = 3) -> str:
    if not keywords:
        return "Miscellaneous"
    parts = []
    singles = [k for k in keywords if " " not in k]
    for k in (singles[:n] if len(singles) >= 2 else keywords):
        title = " ".join(w.upper() if w in {"ai", "gpu", "rbi", "gdp", "ipl", "un", "eu", "us", "uk"} else w.capitalize()
                         for w in k.split())
        parts.append(title)
    return " · ".join(parts[:n])


def llm_name(keywords: list[str], sample_titles: list[str]) -> str | None:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        prompt = (
            "Give a short (2-4 words) human-readable news topic name for a cluster of articles.\n"
            f"Keywords: {', '.join(keywords[:10])}\n"
            "Sample headlines:\n- " + "\n- ".join(sample_titles[:5]) +
            "\nReply with ONLY the topic name."
        )
        msg = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=30,
                                     messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text.strip().strip('"')
    except Exception:
        return None


def name_topics(keywords: dict[int, list[str]], titles_by_cluster: dict[int, list[str]],
                use_llm: bool = False) -> dict[int, str]:
    names = {}
    for cid, kws in keywords.items():
        name = llm_name(kws, titles_by_cluster.get(cid, [])) if use_llm else None
        names[cid] = name or rule_based_name(kws)
    return names
