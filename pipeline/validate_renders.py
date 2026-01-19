"""
Validate rendered handwriting samples using a pre-trained OCR model.
Filters out samples where the rendered image doesn't match the ground truth.
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from collections import Counter

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    from PIL import Image
    import numpy as np
except ImportError:
    Image = None
    np = None

from .utils import setup_logging, read_jsonl, write_jsonl


def load_validator(languages: List[str] = None, gpu: bool = False):
    """Load EasyOCR reader for validation."""
    if easyocr is None:
        raise ImportError("easyocr is required: pip install easyocr")
    languages = languages or ["en"]
    return easyocr.Reader(languages, gpu=gpu, verbose=False)


def validate_single_char(
    reader,
    image_path: Path,
    expected_char: str,
    confidence_threshold: float = 0.3,
) -> Tuple[bool, str, float]:
    """
    Validate a single character image.

    Returns:
        (is_valid, predicted_char, confidence)
    """
    try:
        results = reader.readtext(str(image_path), detail=1, paragraph=False)
    except Exception as e:
        logging.warning(f"OCR failed for {image_path}: {e}")
        return False, "", 0.0

    if not results:
        # No text detected at all
        return False, "", 0.0

    # Get the best prediction
    best_text = ""
    best_conf = 0.0
    for bbox, text, conf in results:
        if conf > best_conf:
            best_text = text
            best_conf = conf

    # Normalize for comparison
    predicted = best_text.strip()
    expected = expected_char.strip()

    # Check if prediction matches expected (case-sensitive)
    is_match = predicted == expected

    # Also accept case-insensitive match with lower confidence threshold
    is_case_insensitive_match = predicted.lower() == expected.lower()

    if is_match and best_conf >= confidence_threshold:
        return True, predicted, best_conf
    elif is_case_insensitive_match and best_conf >= confidence_threshold:
        # Accept case-insensitive but log it
        logging.debug(f"Case mismatch: expected '{expected}', got '{predicted}'")
        return True, predicted, best_conf
    else:
        return False, predicted, best_conf


def validate_dataset(
    data_dir: Path,
    reader,
    confidence_threshold: float = 0.3,
    max_samples: Optional[int] = None,
) -> Tuple[List[Path], List[Path], dict]:
    """
    Validate all samples in a dataset directory.

    Returns:
        (valid_samples, invalid_samples, stats)
    """
    valid = []
    invalid = []
    stats = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "no_detection": 0,
        "wrong_char": 0,
        "low_confidence": 0,
        "by_char": Counter(),
        "errors_by_char": Counter(),
    }

    png_files = sorted(data_dir.glob("*.png"))
    if max_samples:
        png_files = png_files[:max_samples]

    for png_path in png_files:
        gt_path = png_path.with_suffix(".gt.txt")
        if not gt_path.exists():
            logging.warning(f"No ground truth for {png_path}")
            continue

        expected = gt_path.read_text(encoding="utf-8").strip()
        if not expected:
            continue

        stats["total"] += 1
        stats["by_char"][expected] += 1

        is_valid, predicted, confidence = validate_single_char(
            reader, png_path, expected, confidence_threshold
        )

        if is_valid:
            valid.append(png_path)
            stats["valid"] += 1
        else:
            invalid.append(png_path)
            stats["invalid"] += 1
            stats["errors_by_char"][expected] += 1

            if not predicted:
                stats["no_detection"] += 1
            elif predicted.lower() != expected.lower():
                stats["wrong_char"] += 1
                logging.info(
                    f"Wrong char: {png_path.name} expected='{expected}' "
                    f"got='{predicted}' conf={confidence:.2f}"
                )
            else:
                stats["low_confidence"] += 1

    return valid, invalid, stats


def filter_dataset(
    data_dir: Path,
    output_dir: Path,
    valid_samples: List[Path],
    copy_files: bool = True,
) -> None:
    """Copy or move valid samples to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, png_path in enumerate(valid_samples, start=1):
        new_id = f"{idx:06d}"

        # Copy PNG
        src_png = png_path
        dst_png = output_dir / f"{new_id}.png"

        # Copy ground truth
        src_gt = png_path.with_suffix(".gt.txt")
        dst_gt = output_dir / f"{new_id}.gt.txt"

        # Copy SVG if exists
        src_svg = png_path.with_suffix(".svg")
        dst_svg = output_dir / f"{new_id}.svg"

        if copy_files:
            shutil.copy2(src_png, dst_png)
            if src_gt.exists():
                shutil.copy2(src_gt, dst_gt)
            if src_svg.exists():
                shutil.copy2(src_svg, dst_svg)
        else:
            shutil.move(src_png, dst_png)
            if src_gt.exists():
                shutil.move(src_gt, dst_gt)
            if src_svg.exists():
                shutil.move(src_svg, dst_svg)


def main():
    parser = argparse.ArgumentParser(
        description="Validate rendered handwriting samples and filter out bad ones."
    )
    parser.add_argument("--data", type=Path, required=True, help="Input data directory")
    parser.add_argument("--out", type=Path, help="Output directory for valid samples")
    parser.add_argument(
        "--confidence", type=float, default=0.3, help="Minimum confidence threshold"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples to validate"
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for OCR")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only report stats, don't copy files"
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    logging.info("Loading OCR model for validation...")
    reader = load_validator(gpu=args.gpu)

    logging.info(f"Validating samples in {args.data}")
    valid, invalid, stats = validate_dataset(
        args.data,
        reader,
        confidence_threshold=args.confidence,
        max_samples=args.max_samples,
    )

    # Print statistics
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    print(f"Total samples:    {stats['total']}")
    print(f"Valid samples:    {stats['valid']} ({100*stats['valid']/max(1,stats['total']):.1f}%)")
    print(f"Invalid samples:  {stats['invalid']} ({100*stats['invalid']/max(1,stats['total']):.1f}%)")
    print(f"  - No detection: {stats['no_detection']}")
    print(f"  - Wrong char:   {stats['wrong_char']}")
    print(f"  - Low conf:     {stats['low_confidence']}")

    print("\nError rate by character:")
    for char in sorted(stats["by_char"].keys()):
        total = stats["by_char"][char]
        errors = stats["errors_by_char"].get(char, 0)
        error_rate = 100 * errors / max(1, total)
        status = "POOR" if error_rate > 50 else "OK" if error_rate < 20 else "WARN"
        print(f"  '{char}': {errors}/{total} errors ({error_rate:.1f}%) [{status}]")

    if args.out and not args.dry_run:
        logging.info(f"Copying {len(valid)} valid samples to {args.out}")
        filter_dataset(args.data, args.out, valid, copy_files=True)
        print(f"\nFiltered dataset saved to: {args.out}")
    elif args.dry_run:
        print("\n[Dry run - no files copied]")


if __name__ == "__main__":
    main()
