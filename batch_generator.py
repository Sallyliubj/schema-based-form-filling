#!/usr/bin/env python3
"""
Generates diverse personas and corresponding values for T4 forms.
"""

import argparse
import json
import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from value_generator import (
    build_prompt, 
    load_fields, 
    request_values, 
    to_full_payload, 
    duplicate_payload,
    validate_generated_values
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PersonaGenerator:
    """Generate diverse personas for T4 form generation."""
    
    FIRST_NAMES = [
        "Alex", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Avery", "Cameron",
        "Jamie", "Quinn", "Blake", "Sage", "River", "Rowan", "Phoenix", "Skyler",
        "Emery", "Finley", "Hayden", "Kendall", "Logan", "Parker", "Peyton", "Reese",
        "Sam", "Sydney", "Teagan", "Drew", "Elliot", "Harley", "Jessie", "Kai",
        "Lane", "Marley", "Nico", "Oakley", "Remy", "Shay", "Tatum", "Val",
        "Wren", "Zion", "Adrian", "Bryce", "Charlie", "Dakota", "Eden", "Frankie"
    ]
    
    LAST_NAMES = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
        "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
        "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
        "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill",
        "Flores", "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell",
        "Mitchell", "Carter", "Roberts", "Patel", "Singh", "Kumar", "Chen", "Wang"
    ]
    
    COMPANIES = [
        "Tech Innovations Inc.", "Maple Leaf Solutions", "Northern Digital Corp.",
        "Canadian Software Systems", "Great Lakes Technology", "Pacific Coast Enterprises",
        "Atlantic Business Solutions", "Prairie Wind Industries", "Rocky Mountain Tech",
        "Boreal Forest Consulting", "Tundra Analytics", "Aurora Development Group",
        "Glacier Point Systems", "Caribou Creek Technologies", "Moose Jaw Innovations",
        "Thunder Bay Solutions", "Niagara Falls Tech", "Hudson Bay Industries",
        "Yukon Territory Systems", "Northwest Passage Corp.", "Confederation Bridge Tech",
        "CN Tower Solutions", "Parliament Hill Consulting", "Rideau Canal Systems",
        "Stanley Park Technologies", "Banff National Enterprises", "Jasper Analytics",
        "Algonquin Park Solutions", "Thousand Islands Tech", "Bay of Fundy Industries"
    ]
    
    CITIES = [
        ("Toronto", "ON", "M4B 1B3"), ("Vancouver", "BC", "V6B 2N9"), 
        ("Montreal", "QC", "H3A 0G4"), ("Calgary", "AB", "T2P 2M5"),
        ("Ottawa", "ON", "K1P 1J1"), ("Edmonton", "AB", "T5J 2R7"),
        ("Mississauga", "ON", "L5B 1M2"), ("Winnipeg", "MB", "R3C 0V8"),
        ("Quebec City", "QC", "G1R 2J6"), ("Hamilton", "ON", "L8P 4R5"),
        ("Brampton", "ON", "L6V 1A1"), ("Surrey", "BC", "V3T 0A3"),
        ("Laval", "QC", "H7S 1Z5"), ("Halifax", "NS", "B3J 1S9"),
        ("London", "ON", "N6A 3K7"), ("Markham", "ON", "L3R 0P2"),
        ("Vaughan", "ON", "L4L 4Y7"), ("Gatineau", "QC", "J8X 3X2"),
        ("Saskatoon", "SK", "S7K 3J7"), ("Longueuil", "QC", "J4K 1A1"),
        ("Burnaby", "BC", "V5H 2E2"), ("Regina", "SK", "S4P 3Y2"),
        ("Richmond", "BC", "V6Y 1A6"), ("Richmond Hill", "ON", "L4C 1B2"),
        ("Oakville", "ON", "L6H 0H3"), ("Burlington", "ON", "L7R 1A6"),
        ("Barrie", "ON", "L4M 3X9"), ("Oshawa", "ON", "L1H 3Z7"),
        ("Sherbrooke", "QC", "J1H 1Z1"), ("Saguenay", "QC", "G7H 3A1")
    ]
    
    PROFESSIONS = [
        "Software Developer", "Data Analyst", "Marketing Manager", "Sales Representative",
        "Accountant", "Project Manager", "Graphic Designer", "Customer Service Representative",
        "Operations Manager", "Business Analyst", "HR Specialist", "Financial Advisor",
        "Web Developer", "Content Writer", "Social Media Manager", "Quality Assurance Analyst",
        "Database Administrator", "Network Administrator", "Technical Support Specialist",
        "Product Manager", "UX Designer", "Digital Marketing Specialist", "Research Analyst",
        "Administrative Assistant", "Executive Assistant", "Office Manager", "Receptionist",
        "Consultant", "Trainer", "Coordinator", "Supervisor", "Team Lead", "Specialist"
    ]
    
    @classmethod
    def generate_sin(cls) -> str:
        """Generate a valid Canadian Social Insurance Number using Luhn algorithm."""
        # First digit cannot be 0 or 8
        valid_first_digits = ['1', '2', '3', '4', '5', '6', '7', '9']
        first_digit = random.choice(valid_first_digits)
        
        # Generate first 8 digits
        sin_digits = [int(first_digit)]
        for _ in range(7):
            sin_digits.append(random.randint(0, 9))
        
        # Calculate check digit using Luhn algorithm
        checksum = 0
        for i in range(8):
            if i % 2 == 1:  # Even positions (0-indexed) get doubled
                doubled = sin_digits[i] * 2
                checksum += doubled if doubled < 10 else doubled - 9
            else:
                checksum += sin_digits[i]
        
        check_digit = (10 - (checksum % 10)) % 10
        sin_digits.append(check_digit)
        
        return ''.join(map(str, sin_digits))
    
    @classmethod
    def generate_persona(cls) -> Dict[str, Union[int, str, float, None]]:
        """Generate a random persona for T4 generation."""
        first_name = random.choice(cls.FIRST_NAMES)
        last_name = random.choice(cls.LAST_NAMES)
        company = random.choice(cls.COMPANIES)
        profession = random.choice(cls.PROFESSIONS)
        city, province, postal_code = random.choice(cls.CITIES)
        
        # Generate employee address
        street_number = random.randint(1, 9999)
        street_names = [
            "Main Street", "King Street", "Queen Street", "Bay Street", "Yonge Street",
            "Bloor Street", "College Street", "Dundas Street", "Richmond Street",
            "Adelaide Street", "Front Street", "Wellington Street", "Elm Street",
            "Oak Avenue", "Maple Avenue", "Pine Avenue", "Cedar Avenue", "Birch Avenue",
            "First Avenue", "Second Avenue", "Third Avenue", "Park Road", "Hill Road",
            "Lake Road", "River Road", "Forest Road", "Garden Road", "Spring Road"
        ]
        street_name = random.choice(street_names)
        
        # Randomly add apartment/unit number
        if random.random() < 0.4:  # 40% chance of apartment
            apt_number = random.randint(1, 999)
            apt_types = ["Apt", "Unit", "Suite"]
            apt_type = random.choice(apt_types)
            employee_address = f"{street_number} {street_name}, {apt_type} {apt_number}, {city}, {province} {postal_code}"
        else:
            employee_address = f"{street_number} {street_name}, {city}, {province} {postal_code}"
        
        # Generate employer address (50% chance of being null)
        employer_address = None
        if random.random() < 0.5:
            emp_street_number = random.randint(1, 9999)
            emp_street_name = random.choice(street_names)
            emp_city, emp_province, emp_postal_code = random.choice(cls.CITIES)
            employer_address = f"{emp_street_number} {emp_street_name}, {emp_city}, {emp_province} {emp_postal_code}"
        
        # Generate employment income (realistic range: $30,000 - $200,000)
        employment_income = round(random.uniform(30000, 200000), 2)
        
        # Generate income tax deducted (approximately 15-35% of employment income)
        tax_rate = random.uniform(0.15, 0.35)
        income_tax_deducted = round(employment_income * tax_rate, 2)
        
        # Generate SIN (90% chance of having one, 10% null)
        sin_number = cls.generate_sin() if random.random() < 0.9 else None
        
        persona = {
            "tax_year": 2025,
            "sin_number": sin_number,
            "employer_name": company,
            "employer_address": employer_address,
            "employee_name": f"{first_name} {last_name}",
            "employee_address": employee_address,
            "employment_income": employment_income,
            "income_tax_deducted": income_tax_deducted
        }
        
        return persona


def generate_single_sample(
    sample_id: int,
    field_definitions: Dict[str, Dict[str, Union[str, bool]]],
    output_dir: Path,
    skip_validation: bool = False
) -> Dict[str, Union[str, bool, Optional[str]]]:
    """Generate a single T4 sample."""
    try:
        # Generate persona (now returns a dictionary)
        persona = PersonaGenerator.generate_persona()
        
        # Convert persona to string for LLM prompt
        persona_str = (
            f"{persona['employee_name']} is a professional working at {persona['employer_name']} "
            f"and lives at {persona['employee_address']}. "
            f"For the {persona['tax_year']} tax year, they earned ${persona['employment_income']:.2f} "
            f"and had ${persona['income_tax_deducted']:.2f} in income tax deducted."
        )
        if persona['sin_number']:
            persona_str += f" Their SIN is {persona['sin_number']}."
        if persona['employer_address']:
            persona_str += f" The employer is located at {persona['employer_address']}."
        
        # Build prompt and request values
        prompt = build_prompt(personal_info=persona_str, fields=field_definitions)
        generated = request_values(prompt)
        payload = to_full_payload(field_definitions, generated)
        
        # Validate if not skipped
        validation_result = None
        if not skip_validation:
            validation_result = validate_generated_values(payload, field_definitions)
        
        # Duplicate fields for form filling
        duplicated_payload = duplicate_payload(payload)
        
        # Save files
        sample_name = f"sample_{sample_id:03d}"
        
        # Save persona info in the new format
        persona_file = output_dir / "values" / f"{sample_name}_persona.json"
        persona_data = {
            "id": sample_id,
            "label": persona
        }
        with open(persona_file, "w") as f:
            json.dump(persona_data, f, indent=4)
        
        # Save values
        values_file = output_dir / "values" / f"{sample_name}_values.json"
        with open(values_file, "w") as f:
            json.dump(duplicated_payload, f, indent=4)
        
        logger.info(f"Generated sample {sample_id:03d}")
        
        return {
            "sample_id": sample_id,
            "success": True,
            "persona": persona,
            "validation_passed": validation_result.is_valid if validation_result else None,
            "validation_errors": validation_result.errors if validation_result else None,
            "validation_warnings": validation_result.warnings if validation_result else None,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Failed to generate sample {sample_id:03d}: {e}")
        return {
            "sample_id": sample_id,
            "success": False,
            "persona": None,
            "validation_passed": None,
            "validation_errors": None,
            "validation_warnings": None,
            "error": str(e)
        }


def create_output_directories(output_dir: Path) -> None:
    """Create necessary output directories."""
    directories = [
        output_dir,
        output_dir / "values",
        output_dir / "samples",  # For future form images
        output_dir / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def generate_batch_samples(
    num_samples: int,
    fields_path: str,
    output_dir: str,
    max_workers: int = 5,
    skip_validation: bool = False
) -> Dict[str, Union[int, List[Dict]]]:
    """Generate multiple T4 samples in parallel."""
    
    # Setup
    output_path = Path(output_dir)
    create_output_directories(output_path)
    
    # Load field definitions
    try:
        field_definitions = load_fields(fields_path)
    except Exception as e:
        logger.error(f"Failed to load fields: {e}")
        return {"success_count": 0, "failed_count": 1, "results": []}
    
    logger.info(f"Starting batch generation of {num_samples} samples...")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Max workers: {max_workers}")
    
    # Generate samples in parallel
    results = []
    success_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_sample = {
            executor.submit(
                generate_single_sample, 
                i + 1, 
                field_definitions, 
                output_path, 
                skip_validation
            ): i + 1 
            for i in range(num_samples)
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
            logger.info(f"Progress: {completed}/{num_samples} ({completed/num_samples*100:.1f}%)")
    
    # Sort results by sample_id
    results.sort(key=lambda x: x["sample_id"])
    
    # Generate summary
    summary = {
        "generation_timestamp": datetime.now().isoformat(),
        "total_samples": num_samples,
        "success_count": success_count,
        "failed_count": failed_count,
        "success_rate": success_count / num_samples * 100,
        "validation_summary": {
            "samples_with_validation": len([r for r in results if r["validation_passed"] is not None]),
            "validation_passed": len([r for r in results if r["validation_passed"] is True]),
            "validation_failed": len([r for r in results if r["validation_passed"] is False])
        },
        "results": results
    }
    
    # Save summary
    summary_file = output_path / "generation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"Batch generation completed!")
    logger.info(f"Success: {success_count}/{num_samples} ({success_count/num_samples*100:.1f}%)")
    logger.info(f"Failed: {failed_count}/{num_samples}")
    logger.info(f"Summary saved to: {summary_file}")
    
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate multiple T4 form samples with diverse personas."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)"
    )
    parser.add_argument(
        "--fields",
        type=str,
        default="utils/t4_fields.json",
        help="Path to fields JSON file (default: utils/t4_fields.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="batch_samples",
        help="Output directory for generated samples (default: batch_samples)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of parallel workers (default: 5)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation of generated values"
    )
    return parser.parse_args()


def main() -> int:
    """Main function."""
    args = parse_args()
    
    try:
        summary = generate_batch_samples(
            num_samples=args.num_samples,
            fields_path=args.fields,
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            skip_validation=args.skip_validation
        )
        
        if summary["failed_count"] > 0:
            logger.warning(f"Some samples failed to generate. Check logs for details.")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


"""
python batch_generator.py --num-samples 100 --output-dir samples
"""