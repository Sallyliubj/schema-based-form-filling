# Main entry point for the synthetic document pipeline

import argparse

from src.utils import (
    build_parser,
    setup_logging,
    load_yaml_config,
    extract_form_mappings,
)
from src.value_generation import ValueGenerationPipeline
from src.image_generation import ImageGenerationPipeline
from src.image_perturbation import ImagePerturbationPipeline

def run_pipeline(args: argparse.Namespace) -> None:
    setup_logging()
    
    value_generator = ValueGenerationPipeline(args=args)
    value_generator.run()
    image_generator = ImageGenerationPipeline(args=args)
    image_generator.run()
    image_perturbator = ImagePerturbationPipeline(args=args)
    image_perturbator.run()

if __name__ == "__main__":
    parser = build_parser(description="Synthetic document pipeline")
    args = parser.parse_args()

    # Load YAML configuration and attach convenient mappings to args
    config = load_yaml_config(args.config)
    (
        args.form_to_attributes,
        args.value_filling_forms,
        args.llm_forms,
        args.form_to_reference_images,
        args.form_to_template_image,
        args.form_to_coordinates,
    ) = extract_form_mappings(config)
    args.config_data = config

    run_pipeline(args=args)

# Example usage:
# python main.py --config ./config/forms.yaml --output-dir ./test/results --max-workers 4 --num-persona 2