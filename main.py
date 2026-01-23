# Main entry point for the synthetic document pipeline

import argparse

from src.utils import build_parser
from src.value_generation import ValueGenerationPipeline
from src.image_generation import ImageGenerationPipeline
from src.image_perturbation import ImagePerturbationPipeline

def run_pipeline(args: argparse.Namespace) -> None:
    value_generator = ValueGenerationPipeline(args=args)
    value_generator.run()
    image_generator = ImageGenerationPipeline(args=args)
    image_generator.run()
    image_perturbator = ImagePerturbationPipeline(args=args)
    image_perturbator.run()

if __name__ == "__main__":
    parser = build_parser(description="Synthetic document pipeline")
    args = parser.parse_args()

    run_pipeline(args=args)

# Example usage:
# python main.py --attributes-dir ./examples/attributes --sample-images-dir ./examples/images --coordinates-dir ./examples/coordinates --output-dir ./test/results --max-workers 4 --num-persona 2