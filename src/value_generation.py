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
Generate a realistic and coherent user profile.
Values must be logically consistent with each other.

## User Attributes

{attributes}

## Instructions

1. Generate realistic data (e.g. valid SIN format, real city names, proper postal codes).
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
Generate realistic {form_type} form values for the user based on the context below.

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
