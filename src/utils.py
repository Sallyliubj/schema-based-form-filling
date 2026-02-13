import argparse
import logging
from typing import Any, Dict, List, Tuple

# Optional dependency: PyYAML is required to load the configuration file.
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def setup_logging() -> None:
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Disable verbose Azure logging
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    logging.getLogger("azure.identity").setLevel(logging.WARNING)


def build_parser(description: str = "Synthetic document pipeline") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML configuration file specifying per-form settings (attributes, image assets, coordinates, mode).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        default="results",
        help="Path to the output directory which will contain the generated values and images",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--num-persona",
        type=int,
        default=10,
        help="Number of personas to generate (default: 10)",
    )
    parser.add_argument(
        "--text-model",
        type=str,
        default="gpt-5",
        help="Model to use for text generation (default: gpt-5)",
    )
    parser.add_argument(
        "--image-model",
        type=str,
        default="gpt-5",
        help="Model to use for image generation (default: gpt-5)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "azure"],
        help="Provider to use for generation (default: openai)",
    )
    return parser


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load and minimally validate the YAML configuration.

    Expected structure:
    forms:
      - form_type: t4
        mode: template            # one of: template | llm
        attributes: path/to/t4.json
        image: path/to/t4.jpg     # required and only one image for template mode
        coordinates: path/to/t4_coords.json # required if mode=template
      - form_type: paystub
        mode: llm
        attributes: path/to/paystub.json
        images:                # list of image paths for LLM conditioning
          - path/to/paystub1.jpg
          - path/to/paystub2.jpg
    """
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load --config. Please install it with `pip install pyyaml`."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping (YAML object).")
    if "forms" not in data or not isinstance(data["forms"], list):
        raise ValueError("Config must contain a top-level 'forms' list.")

    for idx, entry in enumerate(data["forms"]):
        if not isinstance(entry, dict):
            raise ValueError(f"forms[{idx}] must be a mapping.")
        if "form_type" not in entry or not entry["form_type"]:
            raise ValueError(f"forms[{idx}] is missing required 'form_type'.")
        if "mode" not in entry or entry["mode"] not in ("template", "llm"):
            raise ValueError(
                f"forms[{idx}] must declare 'mode' as 'template' or 'llm'."
            )
        if "attributes" not in entry:
            raise ValueError(f"forms[{idx}] is missing required 'attributes' path.")
        if entry["mode"] == "template":
            if "image" not in entry or "coordinates" not in entry:
                raise ValueError(
                    f"forms[{idx}] (template) must include 'image' and 'coordinates'."
                )
        if entry["mode"] == "llm":
            if "images" not in entry or not isinstance(entry["images"], list):
                raise ValueError(
                    f"forms[{idx}] (llm) must include 'images' as a list."
                )
    return data


def extract_form_mappings(
    config: Dict[str, Any],
) -> Tuple[
    Dict[str, str],  # form_type -> attributes path
    List[str],       # value_filling_forms (coordinate)
    List[str],       # llm_forms
    Dict[str, List[str]],  # form_type -> images (llm)
    Dict[str, str],        # form_type -> image (template)
    Dict[str, str],        # form_type -> coordinates path (template)
    Dict[str, str],        # form_type -> language (llm)
]:
    """
    Build convenient lookup maps from loaded config.
    """
    form_to_attributes: Dict[str, str] = {}
    value_filling_forms: List[str] = []
    llm_forms: List[str] = []
    form_to_reference_images: Dict[str, List[str]] = {}
    form_to_template_image: Dict[str, str] = {}
    form_to_coordinates: Dict[str, str] = {}
    form_to_language: Dict[str, str] = {}
    for entry in config.get("forms", []):
        form_type = entry["form_type"]
        # Attributes
        form_to_attributes[form_type] = entry["attributes"]
        # Mode-specific fields
        if entry["mode"] == "template":
            if form_type not in value_filling_forms:
                value_filling_forms.append(form_type)
            form_to_template_image[form_type] = entry["image"]
            form_to_coordinates[form_type] = entry["coordinates"]
        elif entry["mode"] == "llm":
            if form_type not in llm_forms:
                llm_forms.append(form_type)
            form_to_reference_images[form_type] = entry.get("images", [])
            form_to_language[form_type] = entry.get("language", "English")
    return (
        form_to_attributes,
        value_filling_forms,
        llm_forms,
        form_to_reference_images,
        form_to_template_image,
        form_to_coordinates,
        form_to_language,
    )
