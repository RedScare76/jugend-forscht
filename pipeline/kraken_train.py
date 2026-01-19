import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Set

from .config import PipelineConfig
from .utils import find_kraken_repo, resolve_path, setup_logging


def _collect_pngs(data_dir: Path, allowed_chars: Optional[Set[str]] = None) -> List[str]:
    png_paths = sorted(data_dir.glob("*.png"))
    if allowed_chars is None:
        return [str(path) for path in png_paths]

    filtered: List[str] = []
    skipped = 0
    missing = 0
    for path in png_paths:
        gt_path = path.with_suffix(".gt.txt")
        if not gt_path.exists():
            missing += 1
            continue
        text = gt_path.read_text(encoding="utf-8").rstrip("\n\r")
        if any(ch not in allowed_chars for ch in text):
            skipped += 1
            continue
        filtered.append(str(path))
    if missing:
        logging.warning("Skipped %d images without gt.txt files", missing)
    if skipped:
        logging.info("Filtered out %d images with disallowed characters", skipped)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Kraken OCR model.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--partition", type=float, default=None)
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable image augmentation.",
    )
    parser.add_argument(
        "--allowed-chars",
        type=str,
        default=None,
        help="Only train on samples containing these characters.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    config = PipelineConfig.from_sources(args.config)

    data_dir = resolve_path(args.data)
    allowed_chars = set(args.allowed_chars) if args.allowed_chars else None
    pngs = _collect_pngs(data_dir, allowed_chars=allowed_chars)
    if not pngs:
        raise SystemExit(f"No PNG files found in {data_dir}")

    logging.info(f"Found {len(pngs)} training samples")

    device = args.device or config.kraken_device
    epochs = args.epochs or config.kraken_epochs
    partition = args.partition or config.kraken_partition
    augment = args.augment if args.augment is not None else False

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Add kraken repo to path if found
    kraken_repo = find_kraken_repo()
    if kraken_repo:
        sys.path.insert(0, str(kraken_repo))

    # Use Kraken's Python API directly to avoid command line length limits
    from kraken.lib.train import KrakenTrainer, RecognitionModel

    logging.info(f"Training with device={device}, epochs={epochs}, augment={augment}")

    # Set up hyper parameters with augmentation setting
    hyper_params = {
        'augment': augment,
    }

    # Create the recognition model
    model = RecognitionModel(
        hyper_params=hyper_params,
        training_data=pngs,
        evaluation_data=None,
        partition=partition,
        format_type='path',
        output=str(out_path),
    )

    # Create trainer and train
    trainer = KrakenTrainer(
        accelerator='gpu' if device == 'cuda' else 'cpu',
        devices=1,
        max_epochs=epochs,
        min_epochs=1,
        enable_progress_bar=True,
    )

    trainer.fit(model)

    # Save the best model
    model.nn.save_model(str(out_path))
    logging.info(f"Model saved to {out_path}")


if __name__ == "__main__":
    main()
