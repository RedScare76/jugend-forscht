"""
Font-based handwriting renderer with augmentation.
Generates training data for OCR using handwriting-style fonts
with realistic augmentations to simulate natural variation.
"""

import argparse
import logging
import math
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import numpy as np

from .utils import (
    setup_logging,
    load_phrases,
    normalize_text,
    write_jsonl,
    resolve_path,
    stable_hash_int,
)
from .config import PipelineConfig


# Default fonts that look like handwriting (will be downloaded if needed)
DEFAULT_FONTS = [
    "Caveat",
    "DancingScript",
    "IndieFlower",
    "ShadowsIntoLight",
    "PatrickHand",
    "Kalam",
    "CoveredByYourGrace",
    "GloriaHallelujah",
    "Handlee",
    "PermanentMarker",
]


def find_system_fonts() -> List[Path]:
    """Find font files on the system."""
    font_dirs = [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
        Path.home() / ".local/share/fonts",
    ]

    fonts = []
    for font_dir in font_dirs:
        if font_dir.exists():
            fonts.extend(font_dir.rglob("*.ttf"))
            fonts.extend(font_dir.rglob("*.otf"))

    return fonts


def find_handwriting_fonts(font_dir: Optional[Path] = None) -> List[Path]:
    """Find handwriting-style fonts."""
    if font_dir and font_dir.exists():
        fonts = list(font_dir.glob("*.ttf")) + list(font_dir.glob("*.otf"))
        if fonts:
            return fonts

    # Look for system fonts with handwriting-like names
    system_fonts = find_system_fonts()
    handwriting_keywords = [
        "caveat", "dancing", "indie", "shadow", "patrick", "kalam",
        "covered", "gloria", "handlee", "marker", "handwrit", "script",
        "cursive", "comic", "journal", "sketch", "scrawl", "messy"
    ]

    matching = []
    for font in system_fonts:
        name_lower = font.stem.lower()
        if any(kw in name_lower for kw in handwriting_keywords):
            matching.append(font)

    return matching if matching else system_fonts[:5]  # Fallback to any fonts


class Augmenter:
    """Applies realistic augmentations to rendered text images."""

    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-3, 3),
        scale_range: Tuple[float, float] = (0.9, 1.1),
        shear_range: Tuple[float, float] = (-0.1, 0.1),
        noise_amount: float = 0.02,
        blur_probability: float = 0.3,
        blur_radius_range: Tuple[float, float] = (0.5, 1.5),
        elastic_alpha: float = 15,
        elastic_sigma: float = 3,
        elastic_probability: float = 0.5,
        thickness_variation: Tuple[int, int] = (-1, 1),
    ):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.shear_range = shear_range
        self.noise_amount = noise_amount
        self.blur_probability = blur_probability
        self.blur_radius_range = blur_radius_range
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.elastic_probability = elastic_probability
        self.thickness_variation = thickness_variation

    def apply(self, image: Image.Image, rng: random.Random) -> Image.Image:
        """Apply random augmentations to an image."""
        # Convert to numpy for some operations
        img_array = np.array(image)

        # Apply elastic distortion (makes text look more hand-drawn)
        if rng.random() < self.elastic_probability:
            img_array = self._elastic_transform(img_array, rng)

        # Back to PIL
        image = Image.fromarray(img_array)

        # Random rotation
        angle = rng.uniform(*self.rotation_range)
        if abs(angle) > 0.5:
            image = image.rotate(
                angle,
                resample=Image.BICUBIC,
                expand=True,
                fillcolor=(255, 255, 255)
            )

        # Random shear
        shear = rng.uniform(*self.shear_range)
        if abs(shear) > 0.02:
            image = self._shear_image(image, shear)

        # Random scale
        scale = rng.uniform(*self.scale_range)
        if abs(scale - 1.0) > 0.02:
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.LANCZOS)

        # Optional blur
        if rng.random() < self.blur_probability:
            radius = rng.uniform(*self.blur_radius_range)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        # Add slight noise
        if self.noise_amount > 0:
            img_array = np.array(image)
            noise = np.random.normal(0, self.noise_amount * 255, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)

        return image

    def _elastic_transform(
        self,
        image: np.ndarray,
        rng: random.Random
    ) -> np.ndarray:
        """Apply elastic deformation to make text look more hand-written."""
        from scipy.ndimage import gaussian_filter, map_coordinates

        shape = image.shape[:2]

        # Random displacement fields
        dx = gaussian_filter(
            (np.random.random(shape) * 2 - 1),
            self.elastic_sigma
        ) * self.elastic_alpha
        dy = gaussian_filter(
            (np.random.random(shape) * 2 - 1),
            self.elastic_sigma
        ) * self.elastic_alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = [
            np.clip(y + dy, 0, shape[0] - 1).astype(np.float32),
            np.clip(x + dx, 0, shape[1] - 1).astype(np.float32),
        ]

        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = map_coordinates(
                    image[:, :, i], indices, order=1, mode='constant', cval=255
                )
            return result
        else:
            return map_coordinates(
                image, indices, order=1, mode='constant', cval=255
            ).astype(np.uint8)

    def _shear_image(self, image: Image.Image, shear: float) -> Image.Image:
        """Apply horizontal shear transformation."""
        width, height = image.size
        xshift = abs(shear) * height
        new_width = width + int(xshift)

        # Affine transformation matrix for shear
        transform_matrix = (1, shear, -xshift if shear > 0 else 0, 0, 1, 0)

        return image.transform(
            (new_width, height),
            Image.AFFINE,
            transform_matrix,
            resample=Image.BICUBIC,
            fillcolor=(255, 255, 255)
        )


class FontRenderer:
    """Renders text using fonts with augmentation."""

    def __init__(
        self,
        fonts: List[Path],
        target_height: int = 80,
        padding: int = 10,
        font_size_range: Tuple[int, int] = (48, 72),
        augmenter: Optional[Augmenter] = None,
        ink_colors: List[Tuple[int, int, int]] = None,
    ):
        self.fonts = fonts
        self.target_height = target_height
        self.padding = padding
        self.font_size_range = font_size_range
        self.augmenter = augmenter or Augmenter()
        self.ink_colors = ink_colors or [
            (0, 0, 0),        # Black
            (25, 25, 25),     # Near black
            (0, 0, 139),      # Dark blue
            (0, 0, 100),      # Darker blue
            (50, 50, 50),     # Dark gray
        ]

        # Pre-load fonts at different sizes
        self._font_cache: Dict[Tuple[Path, int], ImageFont.FreeTypeFont] = {}

    def _get_font(self, font_path: Path, size: int) -> ImageFont.FreeTypeFont:
        """Get a font at a specific size (cached)."""
        key = (font_path, size)
        if key not in self._font_cache:
            try:
                self._font_cache[key] = ImageFont.truetype(str(font_path), size)
            except Exception as e:
                logging.warning(f"Failed to load font {font_path}: {e}")
                # Fallback to default font
                self._font_cache[key] = ImageFont.load_default()
        return self._font_cache[key]

    def render(
        self,
        text: str,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Render text as an image with augmentation."""
        rng = random.Random(seed)

        # Select random font and size
        font_path = rng.choice(self.fonts)
        font_size = rng.randint(*self.font_size_range)
        font = self._get_font(font_path, font_size)

        # Select ink color
        ink_color = rng.choice(self.ink_colors)

        # Calculate text size
        # Create a temporary image to measure text
        temp_img = Image.new("RGB", (1, 1), (255, 255, 255))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Create image with padding
        img_width = text_width + self.padding * 4
        img_height = text_height + self.padding * 4

        image = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        # Draw text centered with slight random offset
        x_offset = self.padding * 2 + rng.randint(-5, 5)
        y_offset = self.padding * 2 + rng.randint(-3, 3) - bbox[1]

        draw.text((x_offset, y_offset), text, font=font, fill=ink_color)

        # Apply augmentation
        image = self.augmenter.apply(image, rng)

        # Crop to content and add consistent padding
        image = self._tight_crop(image)

        # Resize to target height
        if self.target_height > 0 and image.height != self.target_height:
            scale = self.target_height / image.height
            new_width = max(1, int(image.width * scale))
            image = image.resize((new_width, self.target_height), Image.LANCZOS)

        return image

    def _tight_crop(self, image: Image.Image) -> Image.Image:
        """Crop to content with padding."""
        # Convert to grayscale to find content bounds
        gray = image.convert("L")
        gray_array = np.array(gray)

        # Find non-white pixels
        non_white = gray_array < 250

        if not non_white.any():
            return image

        rows = np.any(non_white, axis=1)
        cols = np.any(non_white, axis=0)

        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]

        if len(row_indices) == 0 or len(col_indices) == 0:
            return image

        top = max(0, row_indices[0] - self.padding)
        bottom = min(image.height, row_indices[-1] + self.padding)
        left = max(0, col_indices[0] - self.padding)
        right = min(image.width, col_indices[-1] + self.padding)

        return image.crop((left, top, right, bottom))


def render_dataset(
    phrases: List[str],
    output_dir: Path,
    renderer: FontRenderer,
    samples_per_phrase: int = 1,
    seed: int = 42,
    normalization: str = "NFD",
) -> List[Dict[str, Any]]:
    """Render a full dataset."""
    from tqdm import tqdm

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    sample_idx = 0

    for phrase_idx, phrase in enumerate(tqdm(phrases, desc="Rendering", unit="phrase")):
        for sample_num in range(samples_per_phrase):
            sample_idx += 1
            sample_id = f"{sample_idx:06d}"

            # Generate deterministic seed for this sample
            sample_seed = stable_hash_int("font_render", seed, phrase_idx, sample_num, phrase)

            # Render
            image = renderer.render(phrase, seed=sample_seed)

            # Save files
            png_path = output_dir / f"{sample_id}.png"
            gt_path = output_dir / f"{sample_id}.gt.txt"

            image.save(png_path)

            normalized = normalize_text(phrase, normalization=normalization)
            gt_path.write_text(normalized, encoding="utf-8")

            metadata.append({
                "id": sample_id,
                "text": phrase,
                "normalized": normalized,
                "seed": sample_seed,
                "png_filename": png_path.name,
            })

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Render training data using handwriting fonts with augmentation."
    )
    parser.add_argument("--phrases", type=Path, required=True, help="Input phrases file")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--fonts", type=Path, help="Directory containing font files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Max phrases to render")
    parser.add_argument(
        "--samples-per-phrase",
        type=int,
        default=1,
        help="Number of samples per phrase (different augmentations)",
    )
    parser.add_argument("--height", type=int, default=80, help="Target image height")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable augmentation (for debugging)",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    config = PipelineConfig.from_sources(args.config)

    # Find fonts
    fonts = find_handwriting_fonts(args.fonts)
    if not fonts:
        raise SystemExit(
            "No fonts found. Please install handwriting fonts or specify --fonts directory.\n"
            "You can download fonts from: https://fonts.google.com/?category=Handwriting"
        )

    logging.info(f"Found {len(fonts)} fonts")
    for font in fonts[:5]:
        logging.info(f"  - {font.name}")
    if len(fonts) > 5:
        logging.info(f"  ... and {len(fonts) - 5} more")

    # Load phrases
    phrases = load_phrases(args.phrases)
    if not phrases:
        raise SystemExit("No phrases found")

    if args.limit:
        phrases = phrases[:args.limit]

    logging.info(f"Loaded {len(phrases)} phrases")

    # Create renderer
    augmenter = None if args.no_augment else Augmenter()
    renderer = FontRenderer(
        fonts=fonts,
        target_height=args.height,
        augmenter=augmenter,
    )

    # Render dataset
    output_dir = resolve_path(args.out)
    metadata = render_dataset(
        phrases=phrases,
        output_dir=output_dir,
        renderer=renderer,
        samples_per_phrase=args.samples_per_phrase,
        seed=args.seed,
        normalization=config.normalization,
    )

    # Save metadata
    metadata_path = output_dir / "metadata.jsonl"
    write_jsonl(metadata_path, metadata)

    # Write manifest
    manifest_path = output_dir / "manifest.txt"
    manifest_path.write_text(
        "\n".join(m["png_filename"] for m in metadata) + "\n",
        encoding="utf-8",
    )

    print(f"\nRendered {len(metadata)} samples to {output_dir}")
    print(f"Fonts used: {len(fonts)}")


if __name__ == "__main__":
    main()
