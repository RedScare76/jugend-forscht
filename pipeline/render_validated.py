"""
Render characters/phrases with validation.
Renders each sample multiple times with different styles and only keeps
samples that pass OCR validation.
"""

import argparse
import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from tqdm import tqdm

from .config import PipelineConfig
from .handwriting_client import HandwritingClient
from .validate_renders import load_validator, validate_single_char
from .utils import (
    convert_svg_to_png,
    detect_allowed_chars,
    detect_style_ids,
    find_handwriting_repo,
    load_phrases,
    normalize_text,
    setup_logging,
    stable_hash_int,
    write_jsonl,
    resolve_path,
    to_uint32,
)


@dataclass
class RenderAttempt:
    text: str
    style: int
    bias: float
    seed: int
    png_path: Path
    svg_path: Path
    is_valid: bool = False
    predicted: str = ""
    confidence: float = 0.0


def render_single_attempt(
    client: HandwritingClient,
    text: str,
    style: int,
    bias: float,
    seed: int,
    temp_dir: Path,
    attempt_id: str,
    render_height: int,
    render_padding: int,
) -> Tuple[Path, Path]:
    """Render a single attempt and return paths."""
    svg_path, _ = client.generate(
        text=text,
        style=style,
        bias=bias,
        seed=seed,
        request_id=attempt_id,
    )

    png_path = temp_dir / f"{attempt_id}.png"
    svg_target = temp_dir / f"{attempt_id}.svg"

    if svg_path.resolve() != svg_target.resolve():
        shutil.copy2(svg_path, svg_target)

    convert_svg_to_png(
        svg_target,
        png_path,
        target_height=render_height,
        padding=render_padding,
    )

    return png_path, svg_target


def render_with_validation(
    client: HandwritingClient,
    reader,
    text: str,
    styles: List[int],
    bias_range: Tuple[float, float],
    base_seed: int,
    temp_dir: Path,
    render_height: int,
    render_padding: int,
    max_attempts: int = 10,
    confidence_threshold: float = 0.3,
) -> Optional[RenderAttempt]:
    """
    Render a character/phrase multiple times until we get a valid one.

    Returns the first valid attempt, or None if all attempts failed.
    """
    import random

    rng = random.Random(base_seed)

    for attempt_num in range(max_attempts):
        style = rng.choice(styles)
        bias = rng.uniform(bias_range[0], bias_range[1])
        seed = to_uint32(stable_hash_int("attempt", base_seed, attempt_num, text))
        attempt_id = f"attempt_{base_seed}_{attempt_num}"

        try:
            png_path, svg_path = render_single_attempt(
                client=client,
                text=text,
                style=style,
                bias=bias,
                seed=seed,
                temp_dir=temp_dir,
                attempt_id=attempt_id,
                render_height=render_height,
                render_padding=render_padding,
            )
        except Exception as e:
            logging.debug(f"Render failed for '{text}' attempt {attempt_num}: {e}")
            continue

        # Validate
        is_valid, predicted, confidence = validate_single_char(
            reader, png_path, text, confidence_threshold
        )

        if is_valid:
            return RenderAttempt(
                text=text,
                style=style,
                bias=bias,
                seed=seed,
                png_path=png_path,
                svg_path=svg_path,
                is_valid=True,
                predicted=predicted,
                confidence=confidence,
            )
        else:
            logging.debug(
                f"Validation failed for '{text}' attempt {attempt_num}: "
                f"predicted='{predicted}' conf={confidence:.2f}"
            )
            # Clean up failed attempt
            png_path.unlink(missing_ok=True)
            svg_path.unlink(missing_ok=True)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Render handwriting samples with validation filtering."
    )
    parser.add_argument("--phrases", type=Path, required=True, help="Input phrases file")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Max samples to generate")
    parser.add_argument(
        "--samples-per-phrase",
        type=int,
        default=1,
        help="Number of valid samples to generate per phrase",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=15,
        help="Max render attempts per sample before giving up",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Minimum OCR confidence threshold",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for OCR validation")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    config = PipelineConfig.from_sources(args.config)

    # Load OCR validator
    logging.info("Loading OCR model for validation...")
    reader = load_validator(gpu=args.gpu)

    # Detect styles
    handwriting_repo = find_handwriting_repo()
    if handwriting_repo:
        styles = detect_style_ids(handwriting_repo)
        allowed_chars = detect_allowed_chars(handwriting_repo)
    else:
        logging.warning("Could not detect handwriting repo; using default styles")
        styles = list(range(12))
        allowed_chars = None

    # Load phrases
    phrases = load_phrases(args.phrases)
    if not phrases:
        raise SystemExit("No phrases found")

    if args.limit:
        phrases = phrases[: args.limit]

    # Filter phrases by allowed chars
    if allowed_chars:
        original_count = len(phrases)
        phrases = [p for p in phrases if all(c in allowed_chars for c in p)]
        if len(phrases) < original_count:
            logging.info(
                f"Filtered {original_count - len(phrases)} phrases with unsupported chars"
            )

    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    svg_dir = resolve_path(Path(config.handwriting_output_dir))
    svg_dir.mkdir(parents=True, exist_ok=True)

    # Create handwriting client
    client = HandwritingClient(
        base_url=config.handwriting_api_url,
        output_dir=svg_dir,
        timeout_s=config.handwriting_timeout_s,
    )

    if not client.health():
        logging.warning("Handwriting service health check failed")

    # Process phrases
    metadata = []
    sample_idx = 0
    failed_phrases = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for phrase_idx, phrase in enumerate(tqdm(phrases, desc="Rendering", unit="phrase")):
            for sample_num in range(args.samples_per_phrase):
                base_seed = to_uint32(
                    stable_hash_int("validated", args.seed, phrase_idx, sample_num, phrase)
                )

                result = render_with_validation(
                    client=client,
                    reader=reader,
                    text=phrase,
                    styles=styles,
                    bias_range=(config.bias_min, config.bias_max),
                    base_seed=base_seed,
                    temp_dir=temp_path,
                    render_height=config.render_height,
                    render_padding=config.render_padding,
                    max_attempts=args.max_attempts,
                    confidence_threshold=args.confidence,
                )

                if result is None:
                    logging.warning(f"Failed to generate valid sample for '{phrase}'")
                    failed_phrases.append(phrase)
                    continue

                # Save valid sample
                sample_idx += 1
                sample_id = f"{sample_idx:06d}"

                # Move files to output
                final_png = out_dir / f"{sample_id}.png"
                final_svg = out_dir / f"{sample_id}.svg"
                final_gt = out_dir / f"{sample_id}.gt.txt"

                shutil.copy2(result.png_path, final_png)
                shutil.copy2(result.svg_path, final_svg)

                normalized = normalize_text(phrase, normalization=config.normalization)
                final_gt.write_text(normalized, encoding="utf-8")

                metadata.append(
                    {
                        "id": sample_id,
                        "text": phrase,
                        "normalized": normalized,
                        "style": result.style,
                        "bias": result.bias,
                        "seed": result.seed,
                        "ocr_predicted": result.predicted,
                        "ocr_confidence": result.confidence,
                        "png_filename": final_png.name,
                        "svg_filename": final_svg.name,
                    }
                )

                # Clean up temp files
                result.png_path.unlink(missing_ok=True)
                result.svg_path.unlink(missing_ok=True)

    client.close()

    # Save metadata
    metadata_path = out_dir / "metadata.jsonl"
    write_jsonl(metadata_path, metadata)

    # Report
    print("\n" + "=" * 50)
    print("RENDER WITH VALIDATION COMPLETE")
    print("=" * 50)
    print(f"Total phrases:     {len(phrases)}")
    print(f"Valid samples:     {sample_idx}")
    print(f"Failed phrases:    {len(failed_phrases)}")
    print(f"Success rate:      {100 * sample_idx / max(1, len(phrases) * args.samples_per_phrase):.1f}%")
    print(f"Output directory:  {out_dir}")

    if failed_phrases:
        print(f"\nFailed phrases ({len(failed_phrases)}):")
        for p in failed_phrases[:20]:
            print(f"  '{p}'")
        if len(failed_phrases) > 20:
            print(f"  ... and {len(failed_phrases) - 20} more")


if __name__ == "__main__":
    main()
