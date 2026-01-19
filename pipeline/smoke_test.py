import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end smoke test.")
    parser.add_argument("--out", type=Path, default=Path("data/smoke_test"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    out_base = args.out
    phrases_path = out_base / "phrases.jsonl"
    dataset_dir = out_base / "kraken_training"
    model_out = out_base / "models" / "smoke_ocr.mlmodel"

    run(
        [
            sys.executable,
            "-m",
            "pipeline.openai_phrases",
            "--out",
            str(phrases_path),
            "--n",
            "5",
            "--batch-size",
            "5",
            "--seed",
            str(args.seed),
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "pipeline.render_dataset",
            "--phrases",
            str(phrases_path),
            "--out",
            str(dataset_dir),
            "--seed",
            str(args.seed),
            "--concurrency",
            "1",
            "--limit",
            "5",
        ]
    )

    if not args.skip_train:
        run(
            [
                sys.executable,
                "-m",
                "pipeline.kraken_train",
                "--data",
                str(dataset_dir),
                "--out",
                str(model_out),
                "--device",
                args.device,
                "--epochs",
                "1",
            ]
        )


if __name__ == "__main__":
    main()
