import argparse
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from tenacity import RetryError
from tqdm import tqdm

from .config import PipelineConfig
from .handwriting_client import HandwritingClient
from .utils import (
    convert_svg_to_png,
    detect_allowed_chars,
    detect_style_ids,
    find_handwriting_repo,
    load_phrases,
    normalize_text,
    read_jsonl,
    sample_style_bias,
    setup_logging,
    stable_hash_int,
    write_jsonl,
    resolve_path,
    to_uint32,
)


def _build_tasks(
    phrases: List[str],
    out_dir: Path,
    seed: int,
    styles: List[int],
    bias_min: float,
    bias_max: float,
    normalization: str,
    limit: int | None,
    allowed_chars: set[str] | None,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for idx, phrase in enumerate(phrases, start=1):
        if limit and idx > limit:
            break
        if len(phrase) > 75:
            logging.warning("Skipping phrase longer than 75 chars: %s", phrase)
            continue
        if allowed_chars is not None and any(ch not in allowed_chars for ch in phrase):
            logging.warning("Skipping phrase with unsupported characters: %s", phrase)
            continue
        sample_id = f"{idx:06d}"
        png_path = out_dir / f"{sample_id}.png"
        gt_path = out_dir / f"{sample_id}.gt.txt"
        if png_path.exists() and gt_path.exists():
            continue
        style, bias, style_seed = sample_style_bias(phrase, idx, seed, styles, bias_min, bias_max)
        request_seed = to_uint32(stable_hash_int("request", seed, idx, phrase))
        tasks.append(
            {
                "id": sample_id,
                "text": phrase,
                "normalized": normalize_text(phrase, normalization=normalization),
                "style": style,
                "bias": bias,
                "style_seed": style_seed,
                "request_seed": request_seed,
                "png_path": png_path,
                "gt_path": gt_path,
            }
        )
    return tasks


def _process_sample(
    client: HandwritingClient,
    task: Dict[str, Any],
    out_dir: Path,
    render_height: int,
    render_padding: int,
) -> Dict[str, Any]:
    started = time.time()
    request_id = task["id"]
    svg_path, meta = client.generate(
        text=task["text"],
        style=task["style"],
        bias=task["bias"],
        seed=task["request_seed"],
        request_id=request_id,
    )
    svg_target = out_dir / f"{task['id']}.svg"
    svg_target.parent.mkdir(parents=True, exist_ok=True)
    if svg_path.resolve() != svg_target.resolve():
        shutil.copy2(svg_path, svg_target)
    convert_svg_to_png(
        svg_target,
        task["png_path"],
        target_height=render_height,
        padding=render_padding,
    )
    task["gt_path"].write_text(task["normalized"], encoding="utf-8")
    duration_ms = int((time.time() - started) * 1000)
    return {
        "id": task["id"],
        "text": task["text"],
        "normalized": task["normalized"],
        "style": task["style"],
        "bias": task["bias"],
        "style_seed": task["style_seed"],
        "request_seed": task["request_seed"],
        "svg_filename": svg_target.name,
        "png_filename": task["png_path"].name,
        "duration_ms": duration_ms,
        "handwriting_meta": meta,
    }


def _write_manifest(out_dir: Path) -> None:
    entries = sorted(out_dir.glob("*.png"))
    lines = [path.name for path in entries if (out_dir / f"{path.stem}.gt.txt").exists()]
    manifest = out_dir / "manifest.txt"
    manifest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render handwriting dataset from phrases.")
    parser.add_argument("--phrases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    config = PipelineConfig.from_sources(args.config)

    handwriting_repo = find_handwriting_repo()
    if handwriting_repo:
        styles = detect_style_ids(handwriting_repo)
        allowed_chars = detect_allowed_chars(handwriting_repo)
    else:
        logging.warning("Could not detect handwriting repo; using default styles")
        styles = list(range(12))
        allowed_chars = None

    phrases = load_phrases(args.phrases)
    if not phrases:
        raise SystemExit("No phrases found")

    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_dir = resolve_path(Path(config.handwriting_output_dir))
    svg_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = out_dir / "metadata.jsonl"
    existing_meta_rows = read_jsonl(metadata_path)
    metadata = {row.get("id"): row for row in existing_meta_rows if row.get("id")}

    tasks = _build_tasks(
        phrases=phrases,
        out_dir=out_dir,
        seed=args.seed,
        styles=styles,
        bias_min=config.bias_min,
        bias_max=config.bias_max,
        normalization=config.normalization,
        limit=args.limit,
        allowed_chars=allowed_chars,
    )

    for idx, phrase in enumerate(phrases, start=1):
        if args.limit and idx > args.limit:
            break
        if len(phrase) > 75:
            continue
        sample_id = f"{idx:06d}"
        png_path = out_dir / f"{sample_id}.png"
        gt_path = out_dir / f"{sample_id}.gt.txt"
        if not (png_path.exists() and gt_path.exists()):
            continue
        if sample_id in metadata:
            continue
        style, bias, style_seed = sample_style_bias(
            phrase, idx, args.seed, styles, config.bias_min, config.bias_max
        )
        request_seed = to_uint32(stable_hash_int("request", args.seed, idx, phrase))
        svg_name = f"{sample_id}.svg" if (out_dir / f"{sample_id}.svg").exists() else None
        metadata[sample_id] = {
            "id": sample_id,
            "text": phrase,
            "normalized": normalize_text(phrase, normalization=config.normalization),
            "style": style,
            "bias": bias,
            "style_seed": style_seed,
            "request_seed": request_seed,
            "svg_filename": svg_name,
            "png_filename": png_path.name,
            "duration_ms": None,
            "handwriting_meta": {},
        }

    if not tasks:
        logging.info("All samples already rendered")
        write_jsonl(metadata_path, [metadata[k] for k in sorted(metadata.keys())])
        _write_manifest(out_dir)
        return

    client = HandwritingClient(
        base_url=config.handwriting_api_url,
        output_dir=svg_dir,
        timeout_s=config.handwriting_timeout_s,
    )

    if not client.health():
        logging.warning("Handwriting service health check failed")

    errors = 0
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {
            executor.submit(
                _process_sample,
                client,
                task,
                out_dir,
                config.render_height,
                config.render_padding,
            ): task
            for task in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Render", unit="sample"):
            task = futures[future]
            try:
                result = future.result()
            except RetryError as exc:
                errors += 1
                cause = exc.last_attempt.exception() if exc.last_attempt else exc
                logging.error("Retry failed for %s: %s", task["id"], cause)
                continue
            except Exception as exc:
                errors += 1
                logging.error("Failed for %s: %s", task["id"], exc)
                continue
            metadata[result["id"]] = result

    client.close()

    write_jsonl(metadata_path, [metadata[k] for k in sorted(metadata.keys())])
    _write_manifest(out_dir)
    logging.info("Rendered %d samples (%d errors)", len(tasks) - errors, errors)


if __name__ == "__main__":
    main()
