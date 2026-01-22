"""
Value Generator for Synthetic Document Pipeline

This module generates realistic values for the given attributes of each document.
It follows a sequential pipeline where each form builds upon the user profile
and previously generated form data.

"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Union


from openai import OpenAI
from pydantic import BaseModel, Field

from utils.constant import BASE_ATTRIBUTES_FILE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('value_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FieldValue(BaseModel):
    """A single field with its generated value."""

    field_id: str = Field(..., description="Field identifier from the schema")
    value: Optional[Union[str, int, float]] = Field(
        default=None, description="Generated value, or null if not applicable"
    )


class GeneratedFields(BaseModel):
    """Collection of generated field values."""

    fields: List[FieldValue]


class UserProfile(BaseModel):
    """Structured user profile with all base attributes."""

    sin: str = Field(..., description="Social Insurance Number")
    first_name: str
    last_name: str
    gender: Optional[str] = None
    date_of_birth: str
    marital_status: str
    street_address: str
    city: str
    province_territory: str
    postal_code: str
    residency_status: str
    number_of_dependents: Optional[int] = None
    employment_status: str
    occupation: Optional[str] = None
    employer_name: Optional[str] = None
    homeowner_status: str
    has_investments: bool
    disability_status: Optional[str] = None


GENERATE_USER_PROFILE_PROMPT = """
Generate a realistic and coherent user profile for a Canadian resident.
Values must be logically consistent with each other.

## User Attributes

{attributes}

## Instructions

1. Generate realistic Canadian data (valid SIN format, real city names, proper postal codes).
2. Use null for attributes that do not apply to this individual:
   - If employment_status is 'unemployed', 'student', or 'retired', set occupation and employer_name to null
   - If number_of_dependents is 0 or not applicable, it can be null
   - If disability_status is 'none', it can be null
3. Ensure logical consistency:
   - If homeowner_status is 'owner', they should have sufficient income potential
   - Age should be consistent with employment status and life situation
4. Use diverse but realistic combinations (vary income levels, locations, family situations).
5. Return a structured JSON object matching the UserProfile schema.
"""


GENERATE_FORM_VALUES_PROMPT = """
Generate realistic {form_type} form values for a Canadian resident based on the context below.

## User Profile

{user_profile}

## Previously Generated Data

{previous_data}

## Field Definitions for {form_type}

{field_definitions}

## Instructions

1. Use the user profile and any previously generated data to ensure consistency.
2. All values must be logically derived from or consistent with the user profile.
3. Use null for fields that do not apply to this individual's situation:
   - Optional fields that are not relevant
   - Fields that would be empty on the actual form
   - Quebec-specific fields (QPP) for non-Quebec residents, etc.
4. Keep numeric values as raw numbers without currency symbols or commas.
5. Ensure financial calculations are mathematically correct:
   - Deductions should sum correctly
   - Tax amounts should be reasonable for the income level
   - Dates should be consistent and realistic
6. Output must comply with the GeneratedFields schema.
"""


ACCUMULATE_CONTEXT_PROMPT = """
Based on the user profile and the newly generated {form_type} form data, 
extract key financial information that should be carried forward to subsequent forms.

## User Profile
{user_profile}

## Generated {form_type} Data
{form_data}

## Instructions

Summarize the key financial facts that would affect subsequent tax documents:
- Income amounts (annual, periodic)
- Tax deductions and contributions
- Any new information about the user's financial situation

Keep the summary concise but complete.
"""

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


class ValueGenerationPipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = OpenAI()


    def load_attributes(self, form_type: str = "base") -> Dict[str, Any]:
        """Load attribute definitions for a form type."""
        if form_type == "base":
            path = BASE_ATTRIBUTES_FILE
        else:
            path = os.path.join(self.args.attributes_dir, f"{form_type}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Attributes file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


    def format_attributes_for_prompt(self, attributes: Dict[str, Any]) -> str:
        """Format base attributes list for the prompt."""
        lines = []
        for attr in attributes:
            lines.append(f"- {attr['attribute']}: {attr['description']}")
        return "\n".join(lines)


    def format_field_definitions(self, fields: Dict[str, Dict]) -> str:
        """Format field definitions for the prompt."""
        lines = []
        for field_id, meta in fields.items():
            required = "required" if meta.get("required") else "optional"
            field_type = meta.get("type", "string")
            desc = meta.get("desc", "No description")
            lines.append(f"- {field_id} ({field_type} | {required}): {desc}")
        return "\n".join(lines)


    def fields_to_dict(self, generated: GeneratedFields) -> Dict[str, Any]:
        """Convert GeneratedFields to a simple dictionary."""
        return {field.field_id: field.value for field in generated.fields}


    def duplicate_fields_for_forms(self, form_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Duplicate all field values for T4 and T5 forms with '_copy' suffix.
        
        Args:
            form_type: The type of form ('t4', 't5', etc.)
            data: Original field data
            
        Returns:
            Data with duplicated fields for T4/T5, original data for other forms
        """
        if form_type.lower() not in ['t4', 't5']:
            return data
        
        # Create a copy of the original data
        duplicated_data = data.copy()
        
        # Add duplicated fields with '_copy' suffix
        for field_id, value in data.items():
            duplicated_data[f"{field_id}_copy"] = value
        
        return duplicated_data


    def save_form_values(self, form_type: str, id: int, data: Dict[str, Any]) -> Path:
        """Save form values to individual JSON file."""
        os.makedirs(os.path.join(self.args.output_dir, "values", str(id)), exist_ok=True)
        
        # Duplicate fields for T4 and T5 forms
        final_data = self.duplicate_fields_for_forms(form_type, data)
        output_path = os.path.join(self.args.output_dir, "values", str(id), f"{form_type}.json")
        with open(output_path, "w") as f:
            json.dump(final_data, f, indent=2)
        return output_path

    def generate_batch_user_profiles(self) -> List[UserProfile]:
        
        logger.info(f"Starting batch generation of {self.args.num_persona} personas...")
        logger.info(f"Output directory: {self.args.output_dir}")
        logger.info(f"Max workers: {self.args.max_workers}")
        
        # Generate samples in parallel
        results = []
        
        with ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(self.generate_user_profile): i + 1 
                for i in range(self.args.num_persona)
            }
            
            # Collect results
            for future in as_completed(future_to_sample):
                result = future.result()
                results.append(result)
        return results

    def generate_user_profile(self) -> UserProfile:
        """
        Generate a realistic user profile from base attributes.

        Returns:
            UserProfile: Structured user profile with all base attributes.
        """
        attributes = self.load_attributes("base")
        formatted_attrs = self.format_attributes_for_prompt(attributes)

        prompt = GENERATE_USER_PROFILE_PROMPT.format(attributes=formatted_attrs)

        response = self.client.beta.chat.completions.parse(
            model=self.args.model,
            messages=[
                {
                    "role": "system",
                    "content": "Generate a realistic user profile. Return structured JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format=UserProfile,
        )

        return response.choices[0].message.parsed

    def generate_form_values(
        self,
        form_type: str,
        user_profile: UserProfile,
        previous_data: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        # Load field definitions
        fields = self.load_attributes(form_type)

        # Format user profile
        profile_json = user_profile.model_dump_json(indent=2)

        # Format previous data
        if previous_data:
            prev_data_str = json.dumps(previous_data, indent=2)
        else:
            prev_data_str = "No previous data - this is the first form in the pipeline."

        # Format field definitions
        field_defs = self.format_field_definitions(fields)

        # Build prompt
        prompt = GENERATE_FORM_VALUES_PROMPT.format(
            form_type=form_type.upper(),
            user_profile=profile_json,
            previous_data=prev_data_str,
            field_definitions=field_defs,
        )

        response = self.client.beta.chat.completions.parse(
            model=self.args.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate realistic form values based on the user profile and context. "
                        "Ensure all values are logically consistent and calculations are correct. "
                        "Return structured data matching the GeneratedFields schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format=GeneratedFields,
        )

        generated = response.choices[0].message.parsed
        return self.fields_to_dict(generated)


    def run(self) -> Dict[str, Any]:
        """
        Run the full document generation pipeline.

        Args:
            output_dir: Directory to save individual form value files. Defaults to RESULTS_DIR.

        Returns:
            Dictionary containing user_profile and all generated form data.
        """

        print("=" * 60)
        print("Starting Value Generation Process")
        print("=" * 60)

        attributes_dir = Path(self.args.attributes_dir)
        output_dir = Path(self.args.output_dir)
        
        # The name of each document want to synthetically generate, reading from the attributes directory.
        documents = [f.stem for f in attributes_dir.glob("*.json")]

        # Step 1: Generate base user profile
        print(f"Generating user profile...")
        user_profiles = self.generate_batch_user_profiles()

        # Save user profile
        for idx, user_profile in enumerate(user_profiles):
            profile_path = self.save_form_values(
                "user_profile", idx, user_profile.model_dump()
            )
            print(f"  → Saved to: {profile_path}")

        # Initialize accumulated data
        accumulated_data: Dict[str, Dict[str, Any]] = {}

        # Step 2: Generate values for each document
        for idx, user_profile in enumerate(user_profiles):
            for form_type in documents:
                print(f"Generating {form_type} values for user profile {idx}...")
                document_data = self.generate_form_values(form_type, user_profile, accumulated_data)
                accumulated_data[form_type] = document_data
                file_path = self.save_form_values(form_type, idx, document_data)
                print(f"  → Saved to: {file_path}")


        print("\n" + "=" * 60)
        print("Value Generation Complete!")
        print(f"All generated values saved to: {output_dir}")
        print("=" * 60)

        return accumulated_data
