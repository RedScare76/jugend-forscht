import ast
import json
import logging
import os
import subprocess
import random
import re
import shutil
import unicodedata
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageChops


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    try:
        from rich.logging import RichHandler

        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=[RichHandler(rich_tracebacks=True)],
        )
    except Exception:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(message)s",
        )


def load_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("pyyaml is required to load YAML configs") from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping")
    return data


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def normalize_text(text: str, normalization: str = "NFD") -> str:
    text = unicodedata.normalize(normalization, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stable_hash_int(*parts: Any, mod: Optional[int] = None) -> int:
    hasher = sha256()
    for part in parts:
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"|")
    value = int.from_bytes(hasher.digest()[:8], "little")
    return value % mod if mod else value


def to_uint32(value: int) -> int:
    return value % (2**32)


def sample_style_bias(
    phrase: str,
    index: int,
    seed: int,
    styles: List[int],
    bias_min: float,
    bias_max: float,
) -> Tuple[int, float, int]:
    if not styles:
        styles = list(range(12))
    rng_seed = stable_hash_int("style_bias", seed, index, phrase)
    rng = random.Random(rng_seed)
    style = rng.choice(styles)
    bias = rng.triangular(bias_min, bias_max, bias_max)
    return style, bias, rng_seed


def get_workspace_root() -> Path:
    here = Path(__file__).resolve()
    candidate = here.parents[1]
    if (candidate / "pipeline").exists():
        return candidate
    return Path.cwd()


def find_repo_path(base_dir: Path, markers: List[str]) -> Optional[Path]:
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        if all((child / marker).exists() for marker in markers):
            return child
    return None


def find_handwriting_repo(base_dir: Optional[Path] = None) -> Optional[Path]:
    base_dir = base_dir or get_workspace_root()
    return find_repo_path(base_dir, ["demo.py", "rnn.py", "styles"])


def find_kraken_repo(base_dir: Optional[Path] = None) -> Optional[Path]:
    base_dir = base_dir or get_workspace_root()
    return find_repo_path(base_dir, ["pyproject.toml", "kraken"])


def detect_style_ids(handwriting_repo: Path) -> List[int]:
    styles_dir = handwriting_repo / "styles"
    if not styles_dir.exists():
        return list(range(12))
    style_ids = []
    for path in styles_dir.glob("style-*-chars.npy"):
        name = path.name
        try:
            style_id = int(name.split("-")[1])
        except Exception:
            continue
        style_ids.append(style_id)
    return sorted(set(style_ids))


def detect_allowed_chars(handwriting_repo: Path) -> Optional[set[str]]:
    drawing_path = handwriting_repo / "drawing.py"
    if not drawing_path.exists():
        return None
    try:
        tree = ast.parse(drawing_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "alphabet":
                    try:
                        value = ast.literal_eval(node.value)
                    except Exception:
                        return None
                    if isinstance(value, list):
                        return {ch for ch in value if ch and ch != "\x00"}
    return None


def convert_svg_to_png(
    svg_path: Path,
    png_path: Path,
    target_height: int,
    padding: int,
) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_data = svg_path.read_text(encoding="utf-8")
    png_bytes = None
    try:
        import cairosvg

        png_bytes = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"))
    except Exception:
        inkscape = shutil.which("inkscape")
        if not inkscape:
            raise
        temp_png = png_path.with_suffix(".tmp.png")
        cmd = [
            inkscape,
            str(svg_path),
            "--export-type=png",
            f"--export-filename={temp_png}",
        ]
        subprocess.run(cmd, check=True)
        if not temp_png.exists():
            raise RuntimeError("Inkscape failed to produce output")
        png_bytes = temp_png.read_bytes()
        temp_png.unlink(missing_ok=True)

    image = Image.open(BytesIO(png_bytes)).convert("RGBA")
    image = _tight_crop(image, padding=padding)
    image = _resize_to_height(image, target_height=target_height)
    image = _flatten_on_white(image)
    image.save(png_path)


def _tight_crop(image: Image.Image, padding: int) -> Image.Image:
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    diff = ImageChops.difference(image, background)
    bbox = diff.getbbox()
    if not bbox:
        return image
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.width, right + padding)
    bottom = min(image.height, bottom + padding)
    return image.crop((left, top, right, bottom))


def _resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
    if target_height <= 0:
        return image
    scale = target_height / float(image.height)
    width = max(1, int(image.width * scale))
    return image.resize((width, target_height), Image.LANCZOS)


def _flatten_on_white(image: Image.Image) -> Image.Image:
    background = Image.new("RGB", image.size, (255, 255, 255))
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    background.paste(image, mask=image.split()[3])
    return background


def load_phrases(path: Path) -> List[str]:
    phrases: List[str] = []
    if path.suffix.lower() == ".jsonl":
        for row in read_jsonl(path):
            text = row.get("text") or row.get("phrase")
            if text:
                phrases.append(text)
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    phrases.append(line)
    return phrases


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def resolve_path(path: Path, base_dir: Optional[Path] = None) -> Path:
    if path.is_absolute():
        return path
    base = base_dir or get_workspace_root()
    return (base / path).resolve()
