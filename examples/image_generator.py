"""
Image Generator for Synthetic Document Pipeline

This module generates filled form images from generated values.

Generation Methods:
- T4, T5: Pre-defined filling using coordinate mappings
- Paystub, Property Tax, NOA: LLM-based image generation
"""

import base64
import cv2
import json
import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, Optional, List

from openai import OpenAI


PIPELINE_DIR = Path(__file__).parent
IMAGES_DIR = PIPELINE_DIR / "images"
COORDINATES_DIR = PIPELINE_DIR / "coordinates"
VALUES_DIR = PIPELINE_DIR / "results" / "values"
OUTPUT_DIR = PIPELINE_DIR / "results" / "images"

# Forms that use pre-defined coordinate-based filling
# PREDEFINED_FORMS = ["t4", "t5"]
PREDEFINED_FORMS = []

# Forms that use LLM-based generation
# LLM_FORMS = ["paystub", "property_tax", "noa"]
LLM_FORMS = ["paystub"]

DEFAULT_MODEL = "gpt-5"


class FormFiller:
    """
    Generic form filler that works with coordinate mappings.

    Supports two float field formats:
    - Split mode (T4): separate integer_region and float_region
    - Unified mode (T5): single region for entire number

    The mode is auto-detected from the coordinate file structure.
    """

    def __init__(self, image_path: str, coordinates_path: str):
        self.image_path = image_path
        self.img = cv2.imread(image_path)

        if self.img is None:
            raise ValueError(f"Could not load image: {image_path}")

        with open(coordinates_path, "r") as f:
            self.coordinates = json.load(f)

        self.split_float_mode = self._detect_split_float_mode()
        self.font = self._load_font()

    def _detect_split_float_mode(self) -> bool:
        """Detect if coordinates use split float mode."""
        for field_data in self.coordinates.values():
            if field_data.get("type") == "float":
                if "integer_region" in field_data:
                    return True
                if "region" in field_data:
                    return False
        return False

    def _load_font(self, font_size: int = 20) -> ImageFont.FreeTypeFont:
        """Load a suitable font for form filling."""
        font_paths = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/ArialBold.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

        for path in font_paths:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, font_size)
                except Exception:
                    continue

        return ImageFont.load_default()

    def fill_form(
        self, values: Dict[str, Any], output_path: str, padding: int = 2
    ) -> str:
        """Fill the form with provided values."""
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # White out all fields
        for field_data in self.coordinates.values():
            self._white_out_field(draw, field_data)

        # Fill in values
        filled_count = 0
        for field_id, value in values.items():
            if value is None:
                continue

            if field_id not in self.coordinates:
                continue

            field_data = self.coordinates[field_id]

            if isinstance(value, (int, float)) and field_data.get("type") == "float":
                value = f"{value:.2f}"

            field_type = field_data.get("type", "string")
            if field_type == "float" and self.split_float_mode:
                self._fill_split_float_field(draw, field_data, str(value), padding)
            else:
                self._fill_single_field(draw, field_data, str(value), padding)

            filled_count += 1

        img_result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img_result)
        return output_path

    def _white_out_field(self, draw: ImageDraw.Draw, field_data: Dict) -> None:
        """White out a field based on its structure."""
        field_type = field_data.get("type", "string")

        if field_type == "float" and self.split_float_mode:
            if "integer_region" in field_data:
                reg = field_data["integer_region"]
                draw.rectangle(
                    [(reg["x1"], reg["y1"]), (reg["x2"], reg["y2"])],
                    fill=(255, 255, 255),
                )
            if "float_region" in field_data:
                reg = field_data["float_region"]
                draw.rectangle(
                    [(reg["x1"], reg["y1"]), (reg["x2"], reg["y2"])],
                    fill=(255, 255, 255),
                )
        else:
            if "region" in field_data:
                reg = field_data["region"]
                draw.rectangle(
                    [(reg["x1"], reg["y1"]), (reg["x2"], reg["y2"])],
                    fill=(255, 255, 255),
                )

    def _fill_split_float_field(
        self, draw: ImageDraw.Draw, field_data: Dict, value: str, padding: int
    ) -> None:
        """Fill a split float field (T4-style)."""
        int_reg = field_data["integer_region"]
        dec_reg = field_data["float_region"]

        if "." in value:
            int_part, dec_part = value.split(".")
            dec_part = dec_part[:2].ljust(2, "0")
        else:
            int_part = value
            dec_part = "00"

        bbox = draw.textbbox((0, 0), int_part, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        int_region_height = int_reg["y2"] - int_reg["y1"]
        int_text_x = int_reg["x2"] - text_width - padding
        int_text_y = int_reg["y1"] + (int_region_height - text_height) // 2
        draw.text((int_text_x, int_text_y), int_part, font=self.font, fill=(0, 0, 0))

        dec_region_height = dec_reg["y2"] - dec_reg["y1"]
        dec_text_x = dec_reg["x1"] + padding
        dec_text_y = dec_reg["y1"] + (dec_region_height - text_height) // 2
        draw.text((dec_text_x, dec_text_y), dec_part, font=self.font, fill=(0, 0, 0))

    def _fill_single_field(
        self, draw: ImageDraw.Draw, field_data: Dict, value: str, padding: int
    ) -> None:
        """Fill a single-region field."""
        if "region" not in field_data:
            return

        region = field_data["region"]
        region_height = region["y2"] - region["y1"]

        bbox = draw.textbbox((0, 0), value, font=self.font)
        text_height = bbox[3] - bbox[1]

        text_x = region["x1"] + padding
        text_y = region["y1"] + (region_height - text_height) // 2

        draw.text((text_x, text_y), value, font=self.font, fill=(0, 0, 0))


def generate_form_with_llm(
    form_type: str,
    reference_images_paths: List[Path],
    values: Dict[str, Any],
    output_path: str,
    model: str = DEFAULT_MODEL,
) -> Optional[str]:
    """
    Generate a filled form image using LLM.

    Args:
        form_type: Type of form (paystub, property_tax, noa)
        reference_images_paths: List of paths to the reference images
        values: Dictionary of field values to fill
        output_path: Path to save the generated image
        model: OpenAI model to use

    Returns:
        Path to generated image, or None if failed
    """
    client = OpenAI()

    reference_images_urls = []
    for reference_image_path in reference_images_paths:
        with open(reference_image_path, "rb") as f:
            reference_image_data = base64.b64encode(f.read()).decode("utf-8")
            reference_images_urls.append(f"data:image/jpeg;base64,{reference_image_data}")

    prompt = f"""Generate a {form_type.upper()} form image filled with these values:

{json.dumps(values, indent=2)}

Instructions:
1. Use the provided image(s) as reference(s), but feel free to make variations:
   - Adjust layout, spacing, or field arrangement
   - Change fonts, colors, or styling
   - Modify header/footer designs
   - Add or remove decorative elements
2. The generated form should still be recognizable as a {form_type.upper()} document
3. Fill in all provided values in appropriate locations
4. Leave fields empty if the value is null
5. Use realistic formatting for dates, numbers, and currency
6. Make the form look authentic and professional
"""

    content = [{
        "type": "input_image",
        "image_url": reference_images_url,
    } for reference_images_url in reference_images_urls]
    
    content.append({
        "type": "input_text",
        "text": prompt,
    })

    response = client.responses.create(
        model=model,
        instructions=(
            "You are generating a filled form document based on a reference image. "
            "You may introduce visual variations - different layouts, fonts, colors, "
            "or styling - while keeping it recognizable as the same type of document. "
            "Place the provided values into appropriate fields."
        ),
        input=[
            {
                "role": "user",
                "content": content,
            },
        ],
        tools=[{"type": "image_generation"}],
    )

    image_data = [
        output.result
        for output in response.output
        if output.type == "image_generation_call"
    ]

    if image_data:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(image_data[0]))
        return output_path
    else:
        return None


def get_available_forms() -> List[str]:
    """Get list of forms that have generated values."""
    available = []
    for form_type in PREDEFINED_FORMS + LLM_FORMS:
        values_path = VALUES_DIR / f"{form_type}.json"
        if values_path.exists():
            available.append(form_type)
    return available


def generate_form_image(form_type: str) -> Optional[str]:
    """
    Generate a filled form image for a specific form type.

    Args:
        form_type: Type of form to generate

    Returns:
        Path to generated image, or None if failed
    """
    # Check if values exist
    values_path = VALUES_DIR / f"{form_type}.json"
    if not values_path.exists():
        print(f"⚠ No values found for {form_type} at {values_path}")
        return None

    with open(values_path, "r") as f:
        values = json.load(f)

    # Determine output path
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = str(OUTPUT_DIR / f"{form_type}_filled.png")

    if form_type in PREDEFINED_FORMS:
        # Use coordinate-based filling for T4/T5
        image_path = IMAGES_DIR / f"{form_type}.jpg"
        coord_path = COORDINATES_DIR / f"{form_type}.json"

        if not image_path.exists():
            print(f"✗ Template image not found: {image_path}")
            return None
        if not coord_path.exists():
            print(f"✗ Coordinates not found: {coord_path}")
            return None

        print(f"[{form_type.upper()}] Using pre-defined filling...")
        filler = FormFiller(str(image_path), str(coord_path))
        result = filler.fill_form(values, output_path)
        print(f"  → Saved to: {result}")
        return result

    elif form_type in LLM_FORMS:
        # Use LLM-based generation for other forms with multiple reference images with {form_type} as the prefix
        # go through the images directory and get all files with {form_type} as the prefix
        reference_images = [f for f in os.listdir(IMAGES_DIR) if f.startswith(form_type)]
        reference_images_paths = [IMAGES_DIR / f for f in reference_images]

        if not reference_images:
            print(f"✗ No reference images found for {form_type}")
            return None

        print(f"[{form_type.upper()}] Using LLM-based generation...")
        result = generate_form_with_llm(form_type, reference_images_paths, values, output_path)
        if result:
            print(f"  → Saved to: {result}")
        else:
            print(f"  ✗ LLM generation failed for {form_type}")
        return result

    else:
        print(f"✗ Unknown form type: {form_type}")
        return None


def image_generation_pipeline() -> Dict[str, Optional[str]]:
    """
    Run the image generation pipeline for all available forms.

    Returns:
        Dictionary mapping form_type to output path (or None if failed)
    """
    print("=" * 60)
    print("Starting Image Generation Pipeline")
    print("=" * 60)
    print(f"Values directory: {VALUES_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    available_forms = get_available_forms()

    if not available_forms:
        print("\n✗ No form values found. Run value_generator.py first.")
        return {}

    print(f"\nFound values for: {', '.join(available_forms)}")

    results = {}

    for form_type in available_forms:
        print(
            f"\n[{available_forms.index(form_type) + 1}/{len(available_forms)}] Processing {form_type}..."
        )
        results[form_type] = generate_form_image(form_type)

    print("\n" + "=" * 60)
    print("Image Generation Complete!")
    print("=" * 60)

    # Summary
    success = [k for k, v in results.items() if v is not None]
    failed = [k for k, v in results.items() if v is None]

    print(f"\n✓ Generated: {', '.join(success) if success else 'None'}")
    if failed:
        print(f"✗ Failed: {', '.join(failed)}")

    return results


if __name__ == "__main__":
    image_generation_pipeline()
