import argparse
import cv2
import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class T4FormFiller:
    def __init__(self, image_path, coordinates_json):
        self.image_path = image_path
        self.img = cv2.imread(image_path)

        if self.img is None:
            raise ValueError(f"Could not load image: {image_path}")
        with open(coordinates_json, "r") as f:
            self.coordinates = json.load(f)

        # Load font - try to find Arial/Helvetica (standard for T4 forms)
        self.font = self._load_font()

        print(
            f"✓ Loaded {len(self.coordinates)} field definitions from {coordinates_json}"
        )

    def _load_font(self, font_size=20):
        font_paths = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/ArialBold.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]

        for path in font_paths:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, font_size)
                    font_name = os.path.basename(path)
                    print(f"✓ Using font: {font_name} at {font_size}pt")
                    return font
                except Exception:
                    continue

        # Fallback to default font
        print(
            "⚠ Warning: Using default font. Install Arial Bold or similar for best results."
        )
        return ImageFont.load_default()

    def fill_from_input_file(
        self,
        input_json_path,
        output_path,
    ):
        with open(input_json_path, "r") as f:
            input_data = json.load(f)

        return self.fill_form(
            input_data,
            output_path=output_path,
        )

    def fill_form(
        self,
        values,
        output_path,
        padding=2,
    ):
        # Convert OpenCV image to PIL for better font rendering
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Step 1: White out ALL fields specified in coordinates
        print(f"Whiting out {len(self.coordinates)} fields...")
        for field_id, field_data in self.coordinates.items():
            self._white_out_field(draw, field_data)

        # Step 2: Fill in values for fields that have data
        print("Filling in values...")
        for field_id, value in values.items():
            # Skip None or null values - leave them white
            if value is None:
                continue

            # Skip if field not in coordinates
            if field_id not in self.coordinates:
                print(f"⚠ Warning: {field_id} not found in coordinate mapping")
                continue

            field_data = self.coordinates[field_id]

            # Format numbers to string with 2 decimal places if needed
            if isinstance(value, (int, float)) and field_data["type"] == "float":
                value = f"{value:.2f}"

            # Fill the field
            field_type = field_data["type"]
            if field_type == "float":
                self._fill_decimal_field(draw, field_data, value, padding)
            else:
                self._fill_single_field(draw, field_data, value, padding)

        # Convert back to OpenCV format and save
        img_result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img_result)
        print(f"✓ Filled form saved to: {output_path}")
        return output_path

    def _white_out_field(self, draw, field_data):
        """White out a field based on its type"""
        field_type = field_data["type"]

        if field_type == "float":
            # White out integer region
            int_reg = field_data["integer_region"]
            draw.rectangle(
                [(int_reg["x1"], int_reg["y1"]), (int_reg["x2"], int_reg["y2"])],
                fill=(255, 255, 255),
            )

            # White out float region
            float_reg = field_data["float_region"]
            draw.rectangle(
                [
                    (float_reg["x1"], float_reg["y1"]),
                    (float_reg["x2"], float_reg["y2"]),
                ],
                fill=(255, 255, 255),
            )
        else:
            # White out single region (for single/string fields)
            if "float_region" in field_data:
                region = field_data["float_region"]
            elif "region" in field_data:
                region = field_data["region"]
            else:
                return

            draw.rectangle(
                [(region["x1"], region["y1"]), (region["x2"], region["y2"])],
                fill=(255, 255, 255),
            )

    def _fill_decimal_field(self, draw, field_data, value, padding):
        int_reg = field_data["integer_region"]
        dec_reg = field_data["float_region"]

        # Parse value
        if "." in str(value):
            int_part, dec_part = str(value).split(".")
        else:
            int_part = str(value)
            dec_part = "00"

        # Calculate text position (right-aligned for integer, left-aligned for decimal)
        int_region_height = int_reg["y2"] - int_reg["y1"]

        # Get text size for integer part (for right alignment)
        bbox = draw.textbbox((0, 0), int_part, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position integer text (right-aligned)
        int_text_x = int_reg["x2"] - text_width - padding
        int_text_y = int_reg["y1"] + (int_region_height - text_height) // 2

        # Add integer text
        draw.text(
            (int_text_x, int_text_y),
            int_part,
            font=self.font,
            fill=(0, 0, 0),
        )

        # Position decimal text (left-aligned)
        float_region_height = dec_reg["y2"] - dec_reg["y1"]
        float_text_x = dec_reg["x1"] + padding
        float_text_y = dec_reg["y1"] + (float_region_height - text_height) // 2

        # Add decimal text
        draw.text(
            (float_text_x, float_text_y),
            dec_part,
            font=self.font,
            fill=(0, 0, 0),
        )

    def _fill_single_field(self, draw, field_data, value, padding):
        region = field_data["region"]

        # Calculate text position (centered vertically, left-aligned)
        region_height = region["y2"] - region["y1"]

        # Get text size
        bbox = draw.textbbox((0, 0), str(value), font=self.font)
        text_height = bbox[3] - bbox[1]

        text_x = region["x1"] + padding
        text_y = region["y1"] + (region_height - text_height) // 2

        # Add text
        draw.text(
            (text_x, text_y),
            str(value),
            font=self.font,
            fill=(0, 0, 0),
        )


def main():
    parser = argparse.ArgumentParser(
        description="T4 Form Filler - Fill T4 tax forms using coordinate mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with all required inputs
  python form_filler.py --image t4.png --coordinates t4.json --values input/software_developer_t4.json

  # Using short flags
  python form_filler.py -i t4.png -c t4.json -v input/data.json -o output.png

  # Specify output path
  python form_filler.py -i t4.png -c t4.json -v input/data.json -o output.png
        """,
    )

    # Required arguments
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the T4 form image",
    )
    parser.add_argument(
        "--coordinates",
        type=str,
        required=True,
        help="Path to coordinates JSON file with field mappings",
    )
    parser.add_argument(
        "--values",
        type=str,
        required=True,
        help="Path to the file with values to fill in",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the filled form",
    )
    args = parser.parse_args()

    try:
        filler = T4FormFiller(
            image_path=args.image,
            coordinates_json=args.coordinates,
        )

        # Fill from values file (using default padding)
        print(f"\nFilling form from {args.values}...")
        filler.fill_from_input_file(
            input_json_path=args.values,
            output_path=args.output,
        )

        print(f"\n✓ Done! Check {args.output}")
        print(
            "Note: Fields with null values or not in input file are left white (empty)"
        )

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
