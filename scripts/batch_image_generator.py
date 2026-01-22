#!/usr/bin/env python3
"""
Create filled form images from generated samples.
Takes JSON values and creates filled form images.
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import form filling functionality
from form_filler import T4FormFiller

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_image_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def discover_samples(values_dir: Path) -> List[Tuple[int, Path]]:
    """Discover all sample value files in the values directory."""
    samples = []
    
    if not values_dir.exists():
        logger.error(f"Values directory does not exist: {values_dir}")
        return samples
    
    # Look for sample_XXX_values.json files
    for file_path in values_dir.glob("sample_*_values.json"):
        try:
            # Extract sample number from filename
            filename = file_path.stem  # sample_001_values
            parts = filename.split('_')
            if len(parts) >= 2 and parts[0] == 'sample' and parts[2] == 'values':
                sample_id = int(parts[1])
                samples.append((sample_id, file_path))
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse sample ID from {file_path.name}: {e}")
            continue
    
    # Sort by sample ID
    samples.sort(key=lambda x: x[0])
    return samples


def fill_single_form(
    sample_id: int,
    values_file: Path,
    form_image: Path,
    coordinates_file: Path,
    output_dir: Path,
    degraded_output_dir: Optional[Path] = None
) -> Dict[str, any]:
    """Fill a single form and optionally create degraded version."""
    try:
        # Load values
        with open(values_file, 'r') as f:
            values = json.load(f)
        
        # Create output filename
        sample_name = f"sample_{sample_id:03d}"
        output_file = output_dir / f"{sample_name}_filled.png"
        
        # Fill the form
        filler = T4FormFiller(
            image_path=str(form_image),
            coordinates_json=str(coordinates_file)
        )
        
        # Save values to temporary file for form filler
        temp_values_file = output_file.parent / f"temp_{sample_name}_values.json"
        with open(temp_values_file, 'w') as f:
            json.dump(values, f, indent=4)
        
        try:
            filler.fill_from_input_file(
                input_json_path=str(temp_values_file),
                output_path=str(output_file)
            )
        finally:
            # Clean up temporary file
            if temp_values_file.exists():
                temp_values_file.unlink()
        
        # Create degraded version if requested
        degraded_file = None
        if degraded_output_dir:
            degraded_file = degraded_output_dir / f"{sample_name}_degraded.png"
            try:
                from image_degrader import degrade_image
                degrade_image(str(output_file), str(degraded_file))
            except ImportError:
                logger.warning(f"image_degrader not available, skipping degraded version for sample {sample_id}")
            except Exception as e:
                logger.warning(f"Failed to create degraded version for sample {sample_id}: {e}")
        
        logger.info(f"Generated form image for sample {sample_id:03d}")
        
        return {
            "sample_id": sample_id,
            "success": True,
            "output_file": str(output_file),
            "degraded_file": str(degraded_file) if degraded_file else None,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Failed to generate form image for sample {sample_id:03d}: {e}")
        return {
            "sample_id": sample_id,
            "success": False,
            "output_file": None,
            "degraded_file": None,
            "error": str(e)
        }


def create_output_directories(samples_dir: Path, degraded_dir: Optional[Path] = None) -> None:
    """Create necessary output directories."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    if degraded_dir:
        degraded_dir.mkdir(parents=True, exist_ok=True)


def generate_batch_images(
    input_dir: str,
    form_image: str,
    coordinates_file: str,
    max_workers: int = 3,
    create_degraded: bool = False
) -> Dict[str, any]:
    """Generate form images for all samples in the input directory."""
    
    # Setup paths
    input_path = Path(input_dir)
    values_dir = input_path / "values"
    samples_dir = input_path / "samples"
    degraded_dir = input_path / "degraded_samples" if create_degraded else None
    
    # Create output directories
    create_output_directories(samples_dir, degraded_dir)
    
    # Discover samples
    samples = discover_samples(values_dir)
    if not samples:
        logger.error(f"No sample files found in {values_dir}")
        return {"success_count": 0, "failed_count": 0, "results": []}
    
    logger.info(f"Found {len(samples)} samples to process")
    logger.info(f"Form template: {form_image}")
    logger.info(f"Coordinates: {coordinates_file}")
    logger.info(f"Output directory: {samples_dir}")
    if degraded_dir:
        logger.info(f"Degraded output directory: {degraded_dir}")
    logger.info(f"Max workers: {max_workers}")
    
    # Validate input files
    form_path = Path(form_image)
    coord_path = Path(coordinates_file)
    
    if not form_path.exists():
        logger.error(f"Form image not found: {form_image}")
        return {"success_count": 0, "failed_count": len(samples), "results": []}
    
    if not coord_path.exists():
        logger.error(f"Coordinates file not found: {coordinates_file}")
        return {"success_count": 0, "failed_count": len(samples), "results": []}
    
    # Generate images in parallel
    results = []
    success_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_sample = {
            executor.submit(
                fill_single_form,
                sample_id,
                values_file,
                form_path,
                coord_path,
                samples_dir,
                degraded_dir
            ): sample_id
            for sample_id, values_file in samples
        }
        
        # Collect results
        for future in as_completed(future_to_sample):
            result = future.result()
            results.append(result)
            
            if result["success"]:
                success_count += 1
            else:
                failed_count += 1
            
            # Progress update
            completed = success_count + failed_count
            logger.info(f"Progress: {completed}/{len(samples)} ({completed/len(samples)*100:.1f}%)")
    
    # Sort results by sample_id
    results.sort(key=lambda x: x["sample_id"])
    
    # Generate summary
    summary = {
        "generation_timestamp": datetime.now().isoformat(),
        "input_directory": str(input_path),
        "form_template": form_image,
        "coordinates_file": coordinates_file,
        "total_samples": len(samples),
        "success_count": success_count,
        "failed_count": failed_count,
        "success_rate": success_count / len(samples) * 100 if samples else 0,
        "degraded_images_created": create_degraded,
        "output_directories": {
            "samples": str(samples_dir),
            "degraded_samples": str(degraded_dir) if degraded_dir else None
        },
        "results": results
    }
    
    # Save summary
    summary_file = input_path / "image_generation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"Batch image generation completed!")
    logger.info(f"Success: {success_count}/{len(samples)} ({success_count/len(samples)*100:.1f}%)")
    logger.info(f"Failed: {failed_count}/{len(samples)}")
    logger.info(f"Images saved to: {samples_dir}")
    if degraded_dir:
        logger.info(f"Degraded images saved to: {degraded_dir}")
    logger.info(f"Summary saved to: {summary_file}")
    
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate filled form images from batch-generated sample values."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing generated samples (with values/ subdirectory)"
    )
    parser.add_argument(
        "--form-image",
        type=str,
        required=True,
        help="Path to the blank form image template (e.g., data/t4.png)"
    )
    parser.add_argument(
        "--coordinates",
        type=str,
        required=True,
        help="Path to the coordinates JSON file (e.g., data/t4_coord.json)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum number of parallel workers (default: 3, lower than batch generator to avoid memory issues)"
    )
    parser.add_argument(
        "--create-degraded",
        action="store_true",
        help="Also create degraded versions of the images (requires image_degrader.py)"
    )
    return parser.parse_args()


def main() -> int:
    """Main function."""
    args = parse_args()
    
    try:
        summary = generate_batch_images(
            input_dir=args.input_dir,
            form_image=args.form_image,
            coordinates_file=args.coordinates,
            max_workers=args.max_workers,
            create_degraded=args.create_degraded
        )
        
        if summary["failed_count"] > 0:
            logger.warning(f"Some images failed to generate. Check logs for details.")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Image generation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Batch image generation failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


"""
python batch_image_generator.py \
    --input-dir samples \
    --form-image data/t4.png \
    --coordinates data/t4_coord.json
"""