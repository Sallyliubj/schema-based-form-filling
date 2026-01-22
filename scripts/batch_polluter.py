#!/usr/bin/env python3
"""
Apply pollution effects to image samples.
"""

import argparse
import json
import logging
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from image_polluter import pollute_image_multiple, PollutionMethod

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_pollution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

POLLUTION_PRESETS = {
    "angle": {
        "rotation": {"angle_range": (-45, 45)},
    },
    "lightning": {
        "lighting": {"brightness_range": (0.3, 1.8), "contrast_range": (0.5, 1.6), "add_shadow": True, "shadow_intensity": 0.7},
    },
    "blur_options": {
        "resolution": {"scale_factor": 0.5},
        "motion_blur": {"kernel_size": 8, "direction": "random"},
        "blur": {"blur_kernel_size": 9},
    },
    "noise_options": {
        "stains": {"num_stains": 5, "stain_size_range": (30, 60), "intensity_range": (0.5, 0.8)},
        "moire": {"frequency": 0.1, "intensity": 0.4},
        "noise": {"noise_intensity": 0.25},
    },
}


def get_pollution_type(method: str) -> str:
    """
    Map pollution methods to their category types.
    
    Returns one of: "original", "angle", "lightning", "blur", "noise"
    
    Mappings:
    - original: "original"
    - rotation: "angle"
    - lighting: "lightning"
    - resolution, motion_blur, blur: "blur"
    - stains, moire, noise: "noise"
    """
    if method in ["resolution", "motion_blur", "blur"]:
        return "blur"
    
    if method in ["stains", "moire", "noise"]:
        return "noise"
    
    if method == "rotation":
        return "angle"

    if method == "lighting":
        return "lightning"

    if method == "original":
        return "original"
    
    return method


def discover_images(samples_dir: Path) -> List[Tuple[int, Path]]:
    """Discover all sample image files in the samples directory."""
    samples = []
    
    if not samples_dir.exists():
        logger.error(f"Samples directory does not exist: {samples_dir}")
        return samples
    
    # Look for sample_XXX_filled.png files
    for file_path in samples_dir.glob("sample_*_filled.png"):
        try:
            # Extract sample number from filename
            filename = file_path.stem  # sample_001_filled
            parts = filename.split('_')
            if len(parts) >= 2 and parts[0] == 'sample' and parts[2] == 'filled':
                sample_id = int(parts[1])
                samples.append((sample_id, file_path))
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse sample ID from {file_path.name}: {e}")
            continue
    
    # Sort by sample ID
    samples.sort(key=lambda x: x[0])
    return samples


def generate_5_effects_for_image(
    sample_id: int,
    input_file: Path,
    output_dir: Path,
    input_dir: Path,
) -> List[Dict[str, any]]:
    """
    Generate 5 pollution effects for a single image:
    1. original (no pollution)
    2. angle (rotation)
    3. lightning (lighting with shadows)
    4. blur (random choice of resolution/motion_blur/blur)
    5. noise (random choice of stains/moire/noise)
    
    Output files are numbered sequentially:
    - For sample_id=1: outputs 001.png, 002.png, 003.png, 004.png, 005.png
    - For sample_id=2: outputs 006.png, 007.png, 008.png, 009.png, 010.png
    - etc.
    
    Returns:
        List of result dictionaries for each effect
    """
    results = []
    sample_name = f"{sample_id:03d}"
    
    # Calculate output IDs for this sample (5 outputs per input)
    base_output_id = (sample_id - 1) * 5 + 1
    output_ids = [base_output_id + i for i in range(5)]
    
    # Load persona data once
    persona_file = input_dir / "values" / f"sample_{sample_name}_persona.json"
    persona_data = None
    
    if persona_file.exists():
        with open(persona_file, 'r') as f:
            persona_data = json.load(f)
    else:
        logger.warning(f"Persona file not found for sample {sample_id:03d}: {persona_file}")
        persona_data = {
            "id": sample_id,
            "label": {}
        }
    
    # Original (no pollution)
    try:
        output_id = output_ids[0]
        output_file = output_dir / f"{output_id:03d}.png"
        output_json = output_dir / f"{output_id:03d}.json"
        
        # Copy original file
        shutil.copy2(str(input_file), str(output_file))
        
        metadata = {
            "id": output_id,
            "label": persona_data.get("label", {}),
            "type": get_pollution_type("original")
        }
        with open(output_json, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        results.append({
            "sample_id": sample_id,
            "output_id": output_id,
            "effect": "original",
            "success": True,
            "output_file": str(output_file),
            "output_json": str(output_json),
            "method": None,
            "pollution_type": get_pollution_type("original"),
            "error": None
        })
        logger.info(f"Sample {sample_id:03d} → Output {output_id:03d}: original (copied)")
    except Exception as e:
        logger.error(f"Failed to create original for sample {sample_id:03d}: {e}")
        results.append({
            "sample_id": sample_id,
            "output_id": output_ids[0],
            "effect": "original",
            "success": False,
            "error": str(e)
        })
    
    # Angle (rotation)
    try:
        output_id = output_ids[1]
        output_file = output_dir / f"{output_id:03d}.png"
        output_json = output_dir / f"{output_id:03d}.json"
        
        pollute_image_multiple(
            input_path=str(input_file),
            output_path=str(output_file),
            methods=["rotation"],
            **POLLUTION_PRESETS["angle"]
        )
        
        metadata = {
            "id": output_id,
            "label": persona_data.get("label", {}),
            "type": get_pollution_type("rotation")
        }
        with open(output_json, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        results.append({
            "sample_id": sample_id,
            "output_id": output_id,
            "effect": "angle",
            "success": True,
            "output_file": str(output_file),
            "output_json": str(output_json),
            "method": "rotation",
            "pollution_type": get_pollution_type("rotation"),
            "error": None
        })
        logger.info(f"Sample {sample_id:03d} → Output {output_id:03d}: angle (rotation)")
    except Exception as e:
        logger.error(f"Failed to create angle effect for sample {sample_id:03d}: {e}")
        results.append({
            "sample_id": sample_id,
            "output_id": output_ids[1],
            "effect": "angle",
            "success": False,
            "error": str(e)
        })
    
    # Lightning (lighting with shadows)
    try:
        output_id = output_ids[2]
        output_file = output_dir / f"{output_id:03d}.png"
        output_json = output_dir / f"{output_id:03d}.json"
        
        pollute_image_multiple(
            input_path=str(input_file),
            output_path=str(output_file),
            methods=["lighting"],
            **POLLUTION_PRESETS["lightning"]
        )
        
        metadata = {
            "id": output_id,
            "label": persona_data.get("label", {}),
            "type": get_pollution_type("lighting")
        }
        with open(output_json, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        results.append({
            "sample_id": sample_id,
            "output_id": output_id,
            "effect": "lightning",
            "success": True,
            "output_file": str(output_file),
            "output_json": str(output_json),
            "method": "lighting",
            "pollution_type": get_pollution_type("lighting"),
            "error": None
        })
        logger.info(f"Sample {sample_id:03d} → Output {output_id:03d}: lightning (lighting)")
    except Exception as e:
        logger.error(f"Failed to create lightning effect for sample {sample_id:03d}: {e}")
        results.append({
            "sample_id": sample_id,
            "output_id": output_ids[2],
            "effect": "lightning",
            "success": False,
            "error": str(e)
        })
    
    # Blur (random choice)
    try:
        output_id = output_ids[3]
        output_file = output_dir / f"{output_id:03d}.png"
        output_json = output_dir / f"{output_id:03d}.json"
        
        blur_method = random.choice(["resolution", "motion_blur", "blur"])
        blur_preset = {blur_method: POLLUTION_PRESETS["blur_options"][blur_method]}
        
        pollute_image_multiple(
            input_path=str(input_file),
            output_path=str(output_file),
            methods=[blur_method],
            **blur_preset
        )
        
        metadata = {
            "id": output_id,
            "label": persona_data.get("label", {}),
            "type": get_pollution_type(blur_method)
        }
        with open(output_json, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        results.append({
            "sample_id": sample_id,
            "output_id": output_id,
            "effect": "blur",
            "success": True,
            "output_file": str(output_file),
            "output_json": str(output_json),
            "method": blur_method,
            "pollution_type": get_pollution_type(blur_method),
            "error": None
        })
        logger.info(f"Sample {sample_id:03d} → Output {output_id:03d}: blur ({blur_method})")
    except Exception as e:
        logger.error(f"Failed to create blur effect for sample {sample_id:03d}: {e}")
        results.append({
            "sample_id": sample_id,
            "output_id": output_ids[3],
            "effect": "blur",
            "success": False,
            "error": str(e)
        })
    
    # Noise (random choice)
    try:
        output_id = output_ids[4]
        output_file = output_dir / f"{output_id:03d}.png"
        output_json = output_dir / f"{output_id:03d}.json"
        
        noise_method = random.choice(["stains", "moire", "noise"])
        noise_preset = POLLUTION_PRESETS["noise_options"][noise_method].copy()
        
        # For stains, randomize the number
        if noise_method == "stains":
            noise_preset["num_stains"] = random.randint(1, 8)
        
        noise_preset_dict = {noise_method: noise_preset}
        
        pollute_image_multiple(
            input_path=str(input_file),
            output_path=str(output_file),
            methods=[noise_method],
            **noise_preset_dict
        )
        
        metadata = {
            "id": output_id,
            "label": persona_data.get("label", {}),
            "type": get_pollution_type(noise_method)
        }
        with open(output_json, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        results.append({
            "sample_id": sample_id,
            "output_id": output_id,
            "effect": "noise",
            "success": True,
            "output_file": str(output_file),
            "output_json": str(output_json),
            "method": noise_method,
            "pollution_type": get_pollution_type(noise_method),
            "error": None
        })
        logger.info(f"Sample {sample_id:03d} → Output {output_id:03d}: noise ({noise_method})")
    except Exception as e:
        logger.error(f"Failed to create noise effect for sample {sample_id:03d}: {e}")
        results.append({
            "sample_id": sample_id,
            "output_id": output_ids[4],
            "effect": "noise",
            "success": False,
            "error": str(e)
        })
    
    return results


def pollute_single_image(
    sample_id: int,
    input_file: Path,
    output_dir: Path,
    input_dir: Path,
) -> List[Dict[str, any]]:
    """
    Generate 5 pollution effects for a single image.
    
    Returns:
        List of results for each of the 5 effects
    """
    try:
        return generate_5_effects_for_image(sample_id, input_file, output_dir, input_dir)
    except Exception as e:
        logger.error(f"Failed to process sample {sample_id:03d}: {e}")
        # Return error result for all 5 effects
        return [{
            "sample_id": sample_id,
            "effect": effect,
            "success": False,
            "error": str(e)
        } for effect in ["original", "angle", "lightning", "blur", "noise"]]


def create_output_directory(output_dir: Path) -> None:
    """Create output directory if it doesn't exist."""
    output_dir.mkdir(parents=True, exist_ok=True)


def batch_pollute_images(
    input_dir: str,
    output_dir: str,
    max_workers: int = 4
) -> Dict[str, any]:
    """Generate 5 pollution effects for each image in the input directory."""
    
    input_path = Path(input_dir)
    samples_dir = input_path / "samples"
    output_path = Path(output_dir)
    
    create_output_directory(output_path)
    
    images = discover_images(samples_dir)
    if not images:
        logger.error(f"No image files found in {samples_dir}")
        return {"success_count": 0, "failed_count": 0, "results": []}
    
    logger.info(f"Found {len(images)} images to process")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Each image will generate 5 effects: original, angle, lightning, blur, noise")
    logger.info(f"Max workers: {max_workers}")
    
    # Process images in parallel
    all_results = []
    images_processed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {
            executor.submit(
                pollute_single_image,
                sample_id,
                image_file,
                output_path,
                input_path,
            ): sample_id
            for sample_id, image_file in images
        }
        
        # Collect results
        for future in as_completed(future_to_sample):
            effect_results = future.result()  # Returns list of 5 results
            all_results.extend(effect_results)
            images_processed += 1
            
            # Progress update
            logger.info(f"Progress: {images_processed}/{len(images)} images ({images_processed/len(images)*100:.1f}%)")
    
    # Sort results by sample_id and effect
    all_results.sort(key=lambda x: (x["sample_id"], x.get("effect", "")))
    
    # Analyze statistics
    success_count = sum(1 for r in all_results if r["success"])
    failed_count = sum(1 for r in all_results if not r["success"])
    
    effect_counts = {"original": 0, "angle": 0, "lightning": 0, "blur": 0, "noise": 0}
    effect_success = {"original": 0, "angle": 0, "lightning": 0, "blur": 0, "noise": 0}
    type_counts = {}
    method_counts = {}
    
    for result in all_results:
        if result["success"]:
            effect = result.get("effect")
            if effect:
                effect_counts[effect] = effect_counts.get(effect, 0) + 1
                effect_success[effect] = effect_success.get(effect, 0) + 1
            
            pollution_type = result.get("pollution_type")
            if pollution_type:
                type_counts[pollution_type] = type_counts.get(pollution_type, 0) + 1
            
            method = result.get("method")
            if method:
                method_counts[method] = method_counts.get(method, 0) + 1
    

    summary = {
        "generation_timestamp": datetime.now().isoformat(),
        "input_directory": str(input_path),
        "output_directory": str(output_path),
        "method_selection": "5_fixed_effects",
        "total_input_images": len(images),
        "total_output_images": len(images) * 5,
        "success_count": success_count,
        "failed_count": failed_count,
        "success_rate": success_count / len(all_results) * 100 if all_results else 0,
        "pollution_statistics": {
            "effect_distribution": effect_counts,
            "effect_success": effect_success,
            "type_distribution": type_counts,
            "method_distribution": method_counts,
        },
        "results": all_results
    }
    

    summary_file = output_path / "pollution_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"\nBatch pollution completed!")
    logger.info(f"Input images: {len(images)}")
    logger.info(f"Output images: {len(images) * 5} (5 per input)")
    logger.info(f"Success: {success_count}/{len(all_results)} ({success_count/len(all_results)*100:.1f}%)")
    logger.info(f"Failed: {failed_count}/{len(all_results)}")
    logger.info(f"Polluted images saved to: {output_path}")
    logger.info(f"Metadata JSON files saved alongside images")
    logger.info(f"Summary saved to: {summary_file}")
    

    logger.info("\nEffect Distribution (per effect type):")
    for effect, count in sorted(effect_counts.items()):
        success = effect_success.get(effect, 0)
        logger.info(f"  {effect:12}: {success}/{count} successful")
    
    logger.info("\nPollution Type Distribution:")
    for ptype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {ptype:12}: {count} images")
    
    logger.info("\nMethod Distribution (for blur and noise effects):")
    for method, count in sorted(method_counts.items()):
        logger.info(f"  {method:12}: {count} images")
    
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate 5 pollution effects for each batch-generated form image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 5 effects per image
  python batch_polluter.py --input-dir samples --output-dir polluted_samples

For each input image (e.g., sample_001_filled.png), generates 5 sequentially numbered outputs:

  Input: sample_001_filled.png → Outputs: 001.png, 002.png, 003.png, 004.png, 005.png
  Input: sample_002_filled.png → Outputs: 006.png, 007.png, 008.png, 009.png, 010.png
  etc.
"""
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing generated samples (with samples/ subdirectory)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save polluted images"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers (default: 4)"
    )
    return parser.parse_args()


def main() -> int:
    """Main function."""
    args = parse_args()
    
    try:
        summary = batch_pollute_images(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            max_workers=args.max_workers
        )
        
        if summary["failed_count"] > 0:
            logger.warning(f"Some images failed to pollute. Check logs for details.")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Pollution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Batch pollution failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


# Example usage:
# python batch_polluter.py --input-dir samples --output-dir polluted_samples