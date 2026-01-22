"""
Value Generator for Synthetic Document Pipeline

This module generates realistic values for Canadian tax and financial documents.
It follows a sequential pipeline where each form builds upon the user profile
and previously generated form data.

Pipeline Order:
1. Base Profile → Core demographic information
2. Paystub → Employment income details
3. T4 → Annual employment income summary
4. T5 → Investment income (optional based on profile)
5. Property Tax → Property ownership details (optional based on profile)
6. NOA → Final tax assessment aggregating all income sources
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI
from pydantic import BaseModel, Field


# Configuration
# DEFAULT_MODEL = "gpt-4.1"
DEFAULT_MODEL = "gpt-5"
ATTRIBUTES_DIR = Path(__file__).parent / "attributes"
RESULTS_DIR = Path(__file__).parent / "results" / "values"

client = OpenAI()


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


def load_attributes(form_type: str = "base") -> Union[List, Dict]:
    """Load attribute definitions for a form type."""
    path = ATTRIBUTES_DIR / f"{form_type}.json"
    if not path.exists():
        raise FileNotFoundError(f"Attributes file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def format_attributes_for_prompt(attributes: List[Dict]) -> str:
    """Format base attributes list for the prompt."""
    lines = []
    for attr in attributes:
        lines.append(f"- {attr['attribute']}: {attr['description']}")
    return "\n".join(lines)


def format_field_definitions(fields: Dict[str, Dict]) -> str:
    """Format field definitions for the prompt."""
    lines = []
    for field_id, meta in fields.items():
        required = "required" if meta.get("required") else "optional"
        field_type = meta.get("type", "string")
        desc = meta.get("desc", "No description")
        lines.append(f"- {field_id} ({field_type} | {required}): {desc}")
    return "\n".join(lines)


def fields_to_dict(generated: GeneratedFields) -> Dict[str, Any]:
    """Convert GeneratedFields to a simple dictionary."""
    return {field.field_id: field.value for field in generated.fields}


def duplicate_fields_for_forms(form_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
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


def save_form_values(form_type: str, data: Dict[str, Any], output_dir: Path) -> Path:
    """Save form values to individual JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Duplicate fields for T4 and T5 forms
    final_data = duplicate_fields_for_forms(form_type, data)
    
    output_path = output_dir / f"{form_type}.json"
    with open(output_path, "w") as f:
        json.dump(final_data, f, indent=2)
    return output_path


def generate_user_profile() -> UserProfile:
    """
    Generate a realistic user profile from base attributes.

    Returns:
        UserProfile: Structured user profile with all base attributes.
    """
    attributes = load_attributes("base")
    formatted_attrs = format_attributes_for_prompt(attributes)

    prompt = GENERATE_USER_PROFILE_PROMPT.format(attributes=formatted_attrs)

    response = client.beta.chat.completions.parse(
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Generate a realistic Canadian user profile. Return structured JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=UserProfile,
    )

    return response.choices[0].message.parsed


def generate_form_values(
    form_type: str,
    user_profile: UserProfile,
    previous_data: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Generate form values for a specific form type.

    Args:
        form_type: Type of form ('paystub', 't4', 't5', 'property_tax', 'noa')
        user_profile: The base user profile
        previous_data: Dictionary of previously generated form data

    Returns:
        Dictionary of field_id -> value mappings
    """
    # Load field definitions
    fields = load_attributes(form_type)

    # Format user profile
    profile_json = user_profile.model_dump_json(indent=2)

    # Format previous data
    if previous_data:
        prev_data_str = json.dumps(previous_data, indent=2)
    else:
        prev_data_str = "No previous data - this is the first form in the pipeline."

    # Format field definitions
    field_defs = format_field_definitions(fields)

    # Build prompt
    prompt = GENERATE_FORM_VALUES_PROMPT.format(
        form_type=form_type.upper(),
        user_profile=profile_json,
        previous_data=prev_data_str,
        field_definitions=field_defs,
    )

    response = client.beta.chat.completions.parse(
        model=DEFAULT_MODEL,
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
    return fields_to_dict(generated)


def value_generation_pipeline(output_dir: Path = RESULTS_DIR) -> Dict[str, Any]:
    """
    Run the full document generation pipeline.

    Args:
        output_dir: Directory to save individual form value files. Defaults to RESULTS_DIR.

    Returns:
        Dictionary containing user_profile and all generated form data.
    """

    print("=" * 60)
    print("Starting Document Generation Pipeline")
    print("=" * 60)
    print(f"Output directory: {output_dir}")

    # Step 1: Generate base user profile
    print("\n[1/6] Generating user profile...")
    user_profile = generate_user_profile()
    print(
        f"  → Generated profile for: {user_profile.first_name} {user_profile.last_name}"
    )
    print(f"  → Occupation: {user_profile.occupation}")
    print(f"  → Location: {user_profile.city}, {user_profile.province_territory}")

    # Save user profile
    profile_path = save_form_values(
        "user_profile", user_profile.model_dump(), output_dir
    )
    print(f"  → Saved to: {profile_path}")

    # Initialize accumulated data
    accumulated_data: Dict[str, Dict[str, Any]] = {}

    # Step 2: Generate Paystub (if employed)
    if user_profile.employment_status in ["employed", "self-employed"]:
        print("\n[2/6] Generating Paystub values...")
        paystub_data = generate_form_values("paystub", user_profile, accumulated_data)
        accumulated_data["paystub"] = paystub_data
        print(f"  → Gross pay: ${paystub_data.get('gross_pay_current', 'N/A')}")
        paystub_path = save_form_values("paystub", paystub_data, output_dir)
        print(f"  → Saved to: {paystub_path}")
    else:
        print("\n[2/6] Skipping Paystub (not employed)")

    # Step 3: Generate T4 (if employed)
    if user_profile.employment_status == "employed":
        print("\n[3/6] Generating T4 values...")
        t4_data = generate_form_values("t4", user_profile, accumulated_data)
        accumulated_data["t4"] = t4_data
        print(f"  → Employment income (Box 14): ${t4_data.get('14', 'N/A')}")
        t4_path = save_form_values("t4", t4_data, output_dir)
        print(f"  → Saved to: {t4_path}")
    else:
        print("\n[3/6] Skipping T4 (not employed)")

    # Step 4: Generate T5 (if has investments)
    if user_profile.has_investments:
        print("\n[4/6] Generating T5 values...")
        t5_data = generate_form_values("t5", user_profile, accumulated_data)
        accumulated_data["t5"] = t5_data
        dividends = t5_data.get("24") or t5_data.get("10") or 0
        print(f"  → Dividend income: ${dividends}")
        t5_path = save_form_values("t5", t5_data, output_dir)
        print(f"  → Saved to: {t5_path}")
    else:
        print("\n[4/6] Skipping T5 (no investments)")

    # Step 5: Generate Property Tax (if homeowner)
    if user_profile.homeowner_status == "owner":
        print("\n[5/6] Generating Property Tax values...")
        property_data = generate_form_values(
            "property_tax", user_profile, accumulated_data
        )
        accumulated_data["property_tax"] = property_data
        print(
            f"  → Property assessed value: ${property_data.get('assessed_value', 'N/A')}"
        )
        print(f"  → Total amount due: ${property_data.get('total_amount_due', 'N/A')}")
        property_path = save_form_values("property_tax", property_data, output_dir)
        print(f"  → Saved to: {property_path}")
    else:
        print("\n[5/6] Skipping Property Tax (not a homeowner)")

    # Step 6: Generate NOA (always - final summary)
    print("\n[6/6] Generating NOA values...")
    noa_data = generate_form_values("noa", user_profile, accumulated_data)
    accumulated_data["noa"] = noa_data
    print(f"  → Total income (Line 15000): ${noa_data.get('15000', 'N/A')}")
    print(f"  → Net income (Line 23600): ${noa_data.get('23600', 'N/A')}")
    print(f"  → Balance due: ${noa_data.get('balance_due', 'N/A')}")
    noa_path = save_form_values("noa", noa_data, output_dir)
    print(f"  → Saved to: {noa_path}")

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print(f"All values saved to: {output_dir}")
    print("=" * 60)

    return {
        "user_profile": user_profile.model_dump(),
        "forms": accumulated_data,
    }


def main() -> int:
    """Main entry point for CLI usage."""
    try:
        value_generation_pipeline()
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
