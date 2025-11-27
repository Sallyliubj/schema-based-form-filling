import cv2
import json
import os
import argparse
from collections import OrderedDict


class T4CoordinateMapper:
    def __init__(self, image_path, fields_path, coord_path):
        self.image_path = image_path
        self.coord_path = coord_path
        self.fields_path = fields_path
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError(f"Could not load image: {image_path}")

        self.img_display = self.img.copy()
        self.coordinates = OrderedDict()
        self.current_field = None
        self.temp_coords = []

        # Field definitions for T4 form
        self.field_definitions = self._get_field_definitions()
        self.field_index = 0

        # Load existing coordinates if available
        if os.path.exists(coord_path):
            self.load_coordinates()

        # UI settings
        self.window_name = "T4 Coordinate Mapper"

    def _get_field_definitions(self):
        with open(self.fields_path, "r") as f:
            fields = json.load(f)
        return [
            (field_id, field_data["desc"], field_data["type"])
            for field_id, field_data in fields.items()
        ]

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for region selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp_coords.append((x, y))

            # Draw a small circle at clicked position
            cv2.circle(self.img_display, (x, y), 3, (0, 255, 0), -1)

            field_id, description, field_type = self.field_definitions[self.field_index]

            if field_type == "float":
                # Float fields need 4 clicks: integer region (2 clicks) + float region (2 clicks)
                if len(self.temp_coords) == 1:
                    print(f"  → Integer top-left recorded at ({x}, {y})")
                    print(f"  → Click BOTTOM-RIGHT of integer region...")
                elif len(self.temp_coords) == 2:
                    # Draw integer region rectangle
                    cv2.rectangle(
                        self.img_display,
                        self.temp_coords[0],
                        self.temp_coords[1],
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        self.img_display,
                        "INT",
                        (self.temp_coords[0][0], self.temp_coords[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                    print(f"  → Integer bottom-right recorded at ({x}, {y})")
                    print(f"  → Now click TOP-LEFT of Float region...")
                elif len(self.temp_coords) == 3:
                    print(f"  → Float top-left recorded at ({x}, {y})")
                    print(f"  → Click BOTTOM-RIGHT of Float region...")
                elif len(self.temp_coords) == 4:
                    # Draw Float region rectangle
                    cv2.rectangle(
                        self.img_display,
                        self.temp_coords[2],
                        self.temp_coords[3],
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        self.img_display,
                        "FLOAT",
                        (self.temp_coords[2][0], self.temp_coords[2][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                    self._save_field_coordinates(field_id, description, field_type)
            else:
                # Single or text fields need 2 clicks for bounding box
                if len(self.temp_coords) == 1:
                    print(f"  → Top-left recorded at ({x}, {y})")
                    print(f"  → Click BOTTOM-RIGHT corner...")
                elif len(self.temp_coords) == 2:
                    # Draw region rectangle
                    cv2.rectangle(
                        self.img_display,
                        self.temp_coords[0],
                        self.temp_coords[1],
                        (0, 255, 0),
                        2,
                    )
                    self._save_field_coordinates(field_id, description, field_type)

            cv2.imshow(self.window_name, self.img_display)

    def _save_field_coordinates(self, field_id, description, field_type):
        """Save the recorded coordinates for a field"""
        if field_type == "float":
            # Integer region: temp_coords[0] to temp_coords[1]
            # Float region: temp_coords[2] to temp_coords[3]
            int_x1, int_y1 = self.temp_coords[0]
            int_x2, int_y2 = self.temp_coords[1]
            float_x1, float_y1 = self.temp_coords[2]
            float_x2, float_y2 = self.temp_coords[3]

            self.coordinates[field_id] = {
                "description": description,
                "type": field_type,
                "integer_region": {
                    "x1": min(int_x1, int_x2),
                    "y1": min(int_y1, int_y2),
                    "x2": max(int_x1, int_x2),
                    "y2": max(int_y1, int_y2),
                },
                "float_region": {
                    "x1": min(float_x1, float_x2),
                    "y1": min(float_y1, float_y2),
                    "x2": max(float_x1, float_x2),
                    "y2": max(float_y1, float_y2),
                },
            }
            print(f"✓ Saved {field_id}:")
            print(f"    INT region: ({int_x1},{int_y1}) to ({int_x2},{int_y2})")
            print(f"    FLOAT region: ({float_x1},{float_y1}) to ({float_x2},{float_y2})")
        else:
            # Single region: temp_coords[0] to temp_coords[1]
            x1, y1 = self.temp_coords[0]
            x2, y2 = self.temp_coords[1]

            self.coordinates[field_id] = {
                "description": description,
                "type": field_type,
                "region": {
                    "x1": min(x1, x2),
                    "y1": min(y1, y2),
                    "x2": max(x1, x2),
                    "y2": max(y1, y2),
                },
            }
            print(f"✓ Saved {field_id}: ({x1},{y1}) to ({x2},{y2})")

        # Reset for next field
        self.temp_coords = []
        self.field_index += 1

        # Draw field label at top-left of first region
        if field_type == "float":
            label_pos = self.coordinates[field_id]["integer_region"]
            label_x = label_pos["x1"]
            label_y = label_pos["y1"] - 20
        else:
            label_pos = self.coordinates[field_id]["region"]
            label_x = label_pos["x1"]
            label_y = label_pos["y1"] - 20

        cv2.putText(
            self.img_display,
            field_id,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

        # Move to next field or finish
        if self.field_index < len(self.field_definitions):
            self._prompt_next_field()
        else:
            print("\n" + "=" * 60)
            print("✓ All fields mapped!")
            print(f"Coordinates saved to: {self.coord_path}")
            self.save_coordinates()

    def _prompt_next_field(self):
        """Display prompt for next field to map"""
        if self.field_index < len(self.field_definitions):
            field_id, description, field_type = self.field_definitions[self.field_index]
            print("\n" + "-" * 60)
            print(f"[{self.field_index + 1}/{len(self.field_definitions)}] {field_id}")
            print(f"Description: {description}")
            print(f"Type: {field_type}")

            if field_type == "float":
                print("→ Click TOP-LEFT corner of INTEGER region")
            else:
                print("→ Click TOP-LEFT corner of the field region")

    def run(self):
        """Start the interactive mapping session"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("=" * 60)
        print("T4 COORDINATE MAPPER")
        print("=" * 60)
        print("\nINSTRUCTIONS:")
        print("- For each field, click to define rectangular regions")
        print("- Single fields: Click TOP-LEFT, then BOTTOM-RIGHT")
        print(
            "- Float fields: Define INTEGER region (2 clicks), then FLOAT region (2 clicks)"
        )
        print("- Press 'u' to undo last entry")
        print("- Press 's' to save and continue later")
        print("- Press 'r' to reset display")
        print("- Press 'ESC' or 'q' to quit")
        print("=" * 60)

        self._prompt_next_field()

        while True:
            cv2.imshow(self.window_name, self.img_display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord("q"):  # ESC or 'q'
                print("\nExiting... Coordinates saved.")
                self.save_coordinates()
                break
            elif key == ord("s"):  # Save
                self.save_coordinates()
                print(f"\n✓ Progress saved to {self.coord_path}")
            elif key == ord("u"):  # Undo
                self._undo_last()
            elif key == ord("r"):  # Reset display
                self.img_display = self.img.copy()
                self._redraw_all_markers()

        cv2.destroyAllWindows()

    def _undo_last(self):
        """Undo the last field entry"""
        if self.coordinates:
            last_key = list(self.coordinates.keys())[-1]
            del self.coordinates[last_key]
            self.field_index = max(0, self.field_index - 1)
            self.temp_coords = []

            # Redraw display
            self.img_display = self.img.copy()
            self._redraw_all_markers()

            print(f"\n↶ Undone: {last_key}")
            self._prompt_next_field()

    def _redraw_all_markers(self):
        """Redraw all recorded coordinate markers"""
        for field_id, data in self.coordinates.items():
            if data["type"] == "float":
                # Draw integer region
                int_reg = data["integer_region"]
                cv2.rectangle(
                    self.img_display,
                    (int_reg["x1"], int_reg["y1"]),
                    (int_reg["x2"], int_reg["y2"]),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    self.img_display,
                    "INT",
                    (int_reg["x1"], int_reg["y1"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )

                # Draw float region
                float_reg = data["float_region"]
                cv2.rectangle(
                    self.img_display,
                    (float_reg["x1"], float_reg["y1"]),
                    (float_reg["x2"], float_reg["y2"]),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    self.img_display,
                    "FLOAT",
                    (float_reg["x1"], float_reg["y1"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )

                # Draw field label
                cv2.putText(
                    self.img_display,
                    field_id,
                    (int_reg["x1"], int_reg["y1"] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )
            else:
                # Draw single region
                reg = data["region"]
                cv2.rectangle(
                    self.img_display,
                    (reg["x1"], reg["y1"]),
                    (reg["x2"], reg["y2"]),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    self.img_display,
                    field_id,
                    (reg["x1"], reg["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

        cv2.imshow(self.window_name, self.img_display)

    def save_coordinates(self):
        """Save coordinates to JSON file"""
        with open(self.coord_path, "w") as f:
            json.dump(self.coordinates, f, indent=2)
        print(f"\n✓ Coordinates saved to: {self.coord_path}")

    def load_coordinates(self):
        """Load coordinates from JSON file"""
        with open(self.coord_path, "r") as f:
            self.coordinates = json.load(f, object_pairs_hook=OrderedDict)

        # Update field index based on loaded data
        self.field_index = len(self.coordinates)
        print(
            f"✓ Loaded {len(self.coordinates)} existing coordinates from {self.coord_path}"
        )

        # Redraw loaded regions
        self._redraw_all_markers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="T4 Form Coordinate Mapper"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to T4 image file",
    )
    parser.add_argument(
        "--fields",
        type=str,
        required=True,
        help="Path to fields JSON file",
    )
    parser.add_argument(
        "--coordinates",
        type=str,
        required=True,
        help="Path to save the output coordinates JSON file",
    )

    args = parser.parse_args()
    mapper = T4CoordinateMapper(args.image, args.fields, args.coordinates)
    mapper.run()
