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
    *, max_attempts: int = 3,
) -> tuple[list[DigestEntry], list[DigestEntry]]:
    if not 1 <= max_attempts <= 10:
        raise ValueError("LLM_SELECTION_MAX_ATTEMPTS must be between 1 and 10")
    unique = {}
    for article in articles:
        unique.setdefault(article_key(article.url), article)
    remaining = list(unique.values())
    top: list[DigestEntry] = []
    mentions: list[DigestEntry] = []
    for stage, requested in enumerate((5, 5, 4), 1):
        if not remaining:
            break
        count = min(requested, len(remaining))
        previous = top + mentions
        print(f"LLM stage {stage}/3: {len(remaining)} candidates, "
              f"{len(previous)} already selected; selecting up to {count}", flush=True)
        selected: list[DigestEntry] = []
        selected_urls: set[str] = set()
        feedback = ""
        for attempt in range(1, max_attempts + 1):
            candidates_list = [a for a in remaining if article_key(a.url) not in selected_urls]
            needed = count - len(selected)
            system_prompt = build_stage_system_prompt(stage, needed, previous + selected)
            if feedback:
                system_prompt += (
                    f"\nПовторная попытка {attempt}/{max_attempts}. {feedback}\n"
                    f"Ранее принятые статьи уже сохранены. Верни только недостающие {needed} "
                    "статей из текущих кандидатов, с точными URL."
                )
            # Transport, configuration and provider errors propagate. Only an
            # invalid selection or malformed JSON triggers a selection retry.
            response = complete(
                stage, system_prompt, build_user_prompt(candidates_list),
                build_stage_schema(needed, mentions=stage == 3),
            )
            try:
                returned = parse_stage_response(response, candidates_list)
                candidates = {article_key(a.url): a for a in candidates_list}
                unknown = duplicates = excess = 0
                for entry in returned:
                    key = article_key(entry.url)
                    if key in selected_urls:
                        duplicates += 1
                    elif key not in candidates:
                        unknown += 1
                    elif len(selected) >= count:
                        excess += 1
                    else:
                        # Source metadata is authoritative; LLM supplies the summary/category.
                        original = candidates[key]
                        entry.url, entry.title, entry.source = original.url, original.title, original.source
                        entry.author, entry.tags, entry.stats = original.author, original.tags, original.stats
                        selected.append(entry)
                        selected_urls.add(key)
                print(f"LLM stage {stage}/3 attempt {attempt}/{max_attempts}: "
                      f"accepted {len(selected)}/{count}; rejected unknown={unknown}, "
                      f"duplicates={duplicates}, excess={excess}", flush=True)
                if stage < 3 and len(selected) != count:
                    raise ValueError(f"expected {count} valid articles, received {len(selected)}")
                if stage == 3 and returned and not selected:
                    raise ValueError("no valid recommendations in a nonempty response")
                break
            except ValueError as error:
                if attempt == max_attempts:
                    raise ValueError(
                        f"LLM stage {stage}/3 failed after {max_attempts} attempts: {error}"
                    ) from error
                feedback = str(error)
                print(f"LLM stage {stage}/3 retry: {feedback}; "
                      f"requesting {count - len(selected)} missing articles", flush=True)
        if stage == 3:
            mentions.extend(selected)
        else:
            top.extend(selected)
        remaining = [article for article in remaining if article_key(article.url) not in selected_urls]
        print(f"LLM stage {stage}/3 selected {len(selected)} articles", flush=True)
    return top, mentions
