"""Sequential 5 + 5 + up to 4 selection with feedback between stages."""

from typing import Any, Callable

from digest.history import article_key
from digest.models import Article, DigestEntry
from digest.llm.prompt import (
    build_stage_schema, build_stage_system_prompt, build_user_prompt, parse_stage_response,
)


def select_in_stages(
    articles: list[Article],
    complete: Callable[[int, str, str, dict[str, Any]], str],
) -> tuple[list[DigestEntry], list[DigestEntry]]:
    remaining = articles[:]
    top: list[DigestEntry] = []
    mentions: list[DigestEntry] = []
    for stage, requested in enumerate((5, 5, 4), 1):
        if not remaining:
            break
        count = min(requested, len(remaining))
        previous = top + mentions
        print(f"LLM stage {stage}/3: {len(remaining)} candidates, "
              f"{len(previous)} already selected; selecting up to {count}", flush=True)
        response = complete(
            stage, build_stage_system_prompt(stage, count, previous),
            build_user_prompt(remaining), build_stage_schema(count, mentions=stage == 3),
        )
        try:
            returned = parse_stage_response(response, remaining)
            candidates = {article_key(article.url): article for article in remaining}
            selected: list[DigestEntry] = []
            selected_urls: set[str] = set()
            for entry in returned:
                key = article_key(entry.url)
                if key in candidates and key not in selected_urls and len(selected) < count:
                    # Preserve canonical source identity rather than model edits
                    # to URL/title/author. Summary and category come from the LLM.
                    original = candidates[key]
                    entry.url, entry.title, entry.source = original.url, original.title, original.source
                    entry.author, entry.tags, entry.stats = original.author, original.tags, original.stats
                    selected.append(entry)
                    selected_urls.add(key)
            if stage < 3 and len(selected) != count:
                raise ValueError(f"expected {count} valid articles, received {len(selected)}")
        except ValueError as error:
            raise ValueError(f"LLM stage {stage}/3: {error}") from error
        if stage == 3:
            mentions.extend(selected)
        else:
            top.extend(selected)
        remaining = [article for article in remaining if article_key(article.url) not in selected_urls]
        print(f"LLM stage {stage}/3 selected {len(selected)} articles", flush=True)
    return top, mentions
