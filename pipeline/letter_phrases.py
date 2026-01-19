import argparse
import logging
from pathlib import Path
from typing import List

from .utils import resolve_path, setup_logging, write_jsonl


def _unique_chars(chars: str) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for ch in chars:
        if ch in seen:
            continue
        seen.add(ch)
        ordered.append(ch)
    return ordered


def generate_letter_phrases(out_path: Path, chars: List[str], count: int) -> None:
    rows = []
    idx = 1
    for _ in range(count):
        for ch in chars:
            rows.append({"id": f"{idx:06d}", "text": ch})
            idx += 1
    write_jsonl(out_path, rows)
    logging.info("Wrote %d phrases for %d characters to %s", len(rows), len(chars), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate single-character training phrases."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--chars",
        type=str,
        default="abcdefABCDEF",
        help="Characters to include; each character gets --count examples.",
    )
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    chars = _unique_chars(args.chars)
    if not chars:
        raise SystemExit("No characters specified")
    if any(not ch.isascii() for ch in chars):
        raise SystemExit("Only ASCII characters are supported for letter phrases")
    if args.count <= 0:
        raise SystemExit("--count must be positive")

    out_path = resolve_path(args.out)
    generate_letter_phrases(out_path, chars, args.count)


if __name__ == "__main__":
    main()
