import argparse


def build_parser(description: str = "Synthetic document pipeline") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--attributes-dir",
        type=str,
        required=True,
        default="attributes",
        help="Path to the directory which contains the attributes of the documents",
    )
    parser.add_argument(
        "--sample-images-dir",
        type=str,
        required=True,
        default="images",
        help="Path to the directory which contains the sample images of the documents",
    )
    parser.add_argument(
        "--coordinates-dir",
        type=str,
        required=False,
        default="coordinates",
        help="Path to the directory which contains the coordinates of the documents. If the structure of the documents need to be maintained, then the coordinates files need to be provided. Otherwise, the documents will be generated through LLM.",
    )
    parser.add_argument(
        "--llm-forms",
        type=list,
        required=False,
        default=["paystub", "property_tax", "noa"],
        help="List of forms that need to be generated through LLM",
    )
    parser.add_argument(
        "--value-filling-forms",
        type=list,
        required=False,
        default=["t4", "t5"],
        help="List of forms that need to be generated through value filling based on the coordinates",
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
        "--model",
        type=str,
        default="gpt-5",
        help="Model to use for generation (default: gpt-5)",
    )
    return parser
