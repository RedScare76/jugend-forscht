import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import List

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from .config import PipelineConfig
from .utils import (
    detect_allowed_chars,
    find_handwriting_repo,
    normalize_text,
    read_jsonl,
    resolve_path,
    setup_logging,
    write_jsonl,
)

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF\u2700-\u27BF]"
)

def _has_all_caps_word(phrase: str) -> bool:
    for token in re.split(r"\s+", phrase.strip()):
        letters = re.sub(r"[^A-Za-z]", "", token)
        if len(letters) >= 2 and letters.isupper():
            return True
    return False


def _validate_phrase(phrase: str, allowed_chars: set[str] | None = None) -> bool:
    if "\n" in phrase or "\r" in phrase:
        return False
    if len(phrase) < 10 or len(phrase) > 75:
        return False
    if not phrase.isascii():
        return False
    if _has_all_caps_word(phrase):
        return False
    if EMOJI_RE.search(phrase):
        return False
    if allowed_chars is not None:
        for ch in phrase:
            if ch not in allowed_chars:
                return False
    return True


def _extract_text(response) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    try:
        return response.output[0].content[0].text
    except Exception as exc:
        raise RuntimeError("Unable to extract text from OpenAI response") from exc


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=20))
def _call_openai(
    client: OpenAI, model: str, batch_size: int, seed: int, allowed_hint: str | None
) -> List[str]:
    system = (
        "You generate short training phrases for handwriting OCR. "
        "Return only the JSON object matching the schema."
    )
    user = (
        "Generate {count} single-line phrases for handwriting training. "
        "Constraints: 10-75 characters, no newlines, ASCII only (no umlauts/diacritics), "
        "avoid all-caps words/acronyms, varied punctuation and capitalization, "
        "include common abbreviations, avoid emojis, avoid personal names/addresses."
    ).format(count=batch_size)
    if allowed_hint:
        user += f" Use only these characters: {allowed_hint}."
    schema = {
        "type": "object",
        "properties": {
            "phrases": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": batch_size,
                "maxItems": batch_size,
            }
        },
        "required": ["phrases"],
        "additionalProperties": False,
    }
    request = dict(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "phrase_batch",
                "schema": schema,
                "strict": True,
            }
        },
        temperature=0.9,
        seed=seed,
    )
    try:
        response = client.responses.create(**request)
    except Exception as exc:
        if "seed" in str(exc).lower():
            request.pop("seed", None)
            response = client.responses.create(**request)
        else:
            raise
    payload = json.loads(_extract_text(response))
    phrases = payload.get("phrases")
    if not isinstance(phrases, list):
        raise ValueError("OpenAI response did not return a phrase list")
    return phrases


def generate_phrases(
    out_path: Path,
    total: int,
    batch_size: int,
    seed: int,
    model: str,
    api_key: str,
    org: str | None,
    project: str | None,
) -> None:
    client = OpenAI(api_key=api_key, organization=org, project=project)
    handwriting_repo = find_handwriting_repo()
    allowed_chars = detect_allowed_chars(handwriting_repo) if handwriting_repo else None
    if allowed_chars is not None:
        allowed_chars = {ch for ch in allowed_chars if ch.isascii()}
    allowed_hint = "".join(sorted(allowed_chars)) if allowed_chars else None

    existing_rows = read_jsonl(out_path)
    existing = []
    seen = set()
    for row in existing_rows:
        text = row.get("text")
        if not text:
            continue
        if not _validate_phrase(text, allowed_chars=allowed_chars):
            logging.warning("Skipping invalid existing phrase: %s", text)
            continue
        norm = normalize_text(text)
        existing.append(row)
        seen.add(norm)

    if len(existing) >= total:
        logging.info("%s already has %d phrases, nothing to do", out_path, len(existing))
        return

    remaining = total - len(existing)
    rows = list(existing)
    progress = tqdm(total=remaining, desc="Phrases", unit="phrase")

    while remaining > 0:
        request_size = min(batch_size, remaining)
        phrases = _call_openai(client, model, request_size, seed, allowed_hint)
        added = 0
        for phrase in phrases:
            phrase = phrase.strip()
            if not _validate_phrase(phrase, allowed_chars=allowed_chars):
                continue
            norm = normalize_text(phrase)
            if norm in seen:
                continue
            idx = len(rows) + 1
            rows.append({"id": f"{idx:06d}", "text": phrase})
            seen.add(norm)
            remaining -= 1
            added += 1
            progress.update(1)
            if remaining == 0:
                break
        if added == 0:
            time.sleep(1)
    progress.close()
    write_jsonl(out_path, rows)
    logging.info("Wrote %d phrases to %s", len(rows), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate phrase list with OpenAI.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    config = PipelineConfig.from_sources(args.config)
    api_key = config.openai_api_key
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")
    model = args.model or config.openai_model

    out_path = resolve_path(args.out)
    generate_phrases(
        out_path=out_path,
        total=args.n,
        batch_size=args.batch_size,
        seed=args.seed,
        model=model,
        api_key=api_key,
        org=config.openai_org,
        project=config.openai_project,
    )


if __name__ == "__main__":
    main()
