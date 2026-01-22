import argparse
import json
import re
import sys
from typing import Dict, List, Optional, Union, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field

DEFAULT_MODEL = "gpt-5"

client = OpenAI()


class FieldValue(BaseModel):
    field_id: str = Field(..., description="Key from utils/fields.json")
    value: Optional[Union[str, int, float]] = Field(
        default=None, description="Value for the field, or null if unknown"
    )


class GeneratedFields(BaseModel):
    fields: List[FieldValue]


class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str]
    warnings: List[str]


def validate_sin(sin: str) -> bool:
    """Validate Canadian Social Insurance Number (SIN)."""
    if not sin or not isinstance(sin, str):
        return False
    
    # Remove spaces and dashes
    sin_digits = re.sub(r'[\s-]', '', sin)
    
    # Must be exactly 9 digits
    if not re.match(r'^\d{9}$', sin_digits):
        return False
    
    # Cannot start with 0 or 8
    if sin_digits[0] in ['0', '8']:
        return False
    
    # Luhn algorithm check
    digits = [int(d) for d in sin_digits]
    checksum = 0
    
    for i in range(8):
        if i % 2 == 1:  # Even positions (0-indexed) get doubled
            doubled = digits[i] * 2
            checksum += doubled if doubled < 10 else doubled - 9
        else:
            checksum += digits[i]
    
    return (10 - (checksum % 10)) % 10 == digits[8]


def validate_canadian_address(address: str) -> bool:
    """Basic validation for Canadian address format."""
    if not address or not isinstance(address, str):
        return False
    
    address = address.strip()
    if len(address) < 10:  # Too short to be a real address
        return False
    
    # Should contain some basic address components
    # Look for postal code pattern (A1A 1A1)
    postal_code_pattern = r'[A-Z]\d[A-Z]\s?\d[A-Z]\d'
    has_postal_code = bool(re.search(postal_code_pattern, address.upper()))
    
    # Should have some numbers (street number)
    has_numbers = bool(re.search(r'\d', address))
    
    return has_postal_code and has_numbers


def validate_province_code(code: str) -> bool:
    """Validate Canadian province/territory employment codes."""
    if not code or not isinstance(code, str):
        return False
    
    valid_codes = {
        '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',  # Provinces
        '20', '21', '22'  # Territories
    }
    
    return code.strip() in valid_codes


def validate_year(year: str) -> bool:
    """Validate tax year."""
    if not year or not isinstance(year, str):
        return False
    
    try:
        year_int = int(year)
        return 2000 <= year_int <= 2030  # Reasonable range
    except ValueError:
        return False



def validate_financial_consistency(payload: Dict[str, Optional[Union[str, int, float]]]) -> List[str]:
    """Check financial value consistency and reasonableness."""
    errors = []
    
    # Get numeric values 
    def get_float(key: str) -> float:
        val = payload.get(key)
        if val is None:
            return 0.0
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0
    
    employment_income = get_float('14')
    cpp_contributions = get_float('16')
    cpp2_contributions = get_float('16A')
    qpp_contributions = get_float('17')
    qpp2_contributions = get_float('17A')
    ei_premiums = get_float('18')
    income_tax = get_float('22')
    ei_insurable = get_float('24')
    cpp_pensionable = get_float('26')
    
    # Employment income should be positive and reasonable
    if employment_income <= 0:
        errors.append("Employment income (box 14) should be positive")
    elif employment_income > 1000000:  # $1M seems like a reasonable upper bound
        errors.append("Employment income (box 14) seems unreasonably high")
    
    # Income tax should not exceed employment income
    if income_tax > employment_income:
        errors.append("Income tax (box 22) cannot exceed employment income (box 14)")
    
    # EI insurable earnings should not exceed employment income
    if ei_insurable > employment_income:
        errors.append("EI insurable earnings (box 24) cannot exceed employment income (box 14)")
    
    # CPP pensionable earnings should not exceed employment income
    if cpp_pensionable > employment_income:
        errors.append("CPP pensionable earnings (box 26) cannot exceed employment income (box 14)")
    
    return errors


def validate_name(name: str) -> bool:
    """Validate person name format."""
    if not name or not isinstance(name, str):
        return False
    
    name = name.strip()
    if len(name) < 1:
        return False
    
    # Should contain only letters, spaces, hyphens, apostrophes
    return bool(re.match(r"^[A-Za-z\s\-']+$", name))


def validate_generated_values(
    payload: Dict[str, Optional[Union[str, int, float]]],
    field_definitions: Dict[str, Dict[str, Union[str, bool]]]
) -> ValidationResult:
    """Validation of generated T4 values."""
    errors = []
    warnings = []
    
    # Validate names
    first_name = payload.get('employee_first_name')
    if first_name is not None and not validate_name(str(first_name)):
        errors.append("Invalid first name format")
    
    last_name = payload.get('employee_last_name')
    if last_name is not None and not validate_name(str(last_name)):
        errors.append("Invalid last name format")
    
    # Validate address
    address = payload.get('employee_address')
    if address is not None and not validate_canadian_address(str(address)):
        warnings.append("Address format may not be valid Canadian format")
    
    # Validate year
    year = payload.get('year')
    if year is not None and not validate_year(str(year)):
        errors.append("Invalid tax year")
    
    # Check required fields
    for field_id, meta in field_definitions.items():
        if meta.get('required', False) and payload.get(field_id) is None:
            errors.append(f"Required field '{field_id}' is missing")
    
    # Validate financial consistency
    financial_errors = validate_financial_consistency(payload)
    errors.extend(financial_errors)
    
    # Check for negative financial values where they shouldn't be
    financial_fields = ['14', '16', '16A', '17', '17A', '18', '20', '22', '24', '26', '34', '38', '40', '44', '46', '52', '55', '56']
    for field_id in financial_fields:
        value = payload.get(field_id)
        if value is not None:
            try:
                float_val = float(value)
                if float_val < 0:
                    errors.append(f"Financial field '{field_id}' cannot be negative")
            except (ValueError, TypeError):
                errors.append(f"Financial field '{field_id}' must be a valid number")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


def load_fields(path: str) -> Dict[str, Dict[str, Union[str, bool]]]:
    with open(path, "r") as f:
        return json.load(f)


def build_prompt(
    personal_info: str,
    fields: Dict[str, Dict[str, Union[str, bool]]],
) -> str:
    field_lines = [
        f"- {field_id} ({meta.get('type')} | {'required' if meta.get('required') else 'optional'}): "
        f"{meta.get('desc', 'No description provided')}"
        for field_id, meta in fields.items()
    ]
    field_block = "\n".join(field_lines)

    return (
        f"Generate realistic T4 tax form values for a Canadian employee.\n"
        f"Personal information: {personal_info}\n"
        f"Field definitions:\n{field_block}\n\n"
        f"Instructions:\n"
        f"1. Use a single coherent scenario based on the provided personal information.\n"
        f"2. Generate realistic values that are consistent with the person's details.\n"
        f"3. Provide a value for every field listed; use null when the person would not have one.\n"
        f"4. Keep numeric values as raw numbers without currency symbols or commas.\n"
        f"5. Output must comply with the provided schema."
    )


def request_values(prompt: str) -> GeneratedFields:
    response = client.responses.parse(
        model=DEFAULT_MODEL,
        instructions=(
            "Return structured data that matches the GeneratedFields schema. "
            "Each field_id should appear exactly once."
        ),
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        text_format=GeneratedFields,
    )
    return response.output_parsed


def to_full_payload(
    field_definitions: Dict[str, Dict[str, Union[str, bool]]],
    generated: GeneratedFields,
) -> Dict[str, Optional[Union[str, int, float]]]:
    payload = {field_id: None for field_id in field_definitions.keys()}
    for field in generated.fields:
        if field.field_id in payload:
            payload[field.field_id] = field.value
        else:
            print(
                f"Warning: Model returned unknown field_id {field.field_id}.",
                file=sys.stderr,
            )
    missing = [k for k, v in payload.items() if v is None]
    if missing:
        print(
            f"Info: Fields without model values defaulted to null: {missing}",
            file=sys.stderr,
        )
    return payload


def duplicate_payload(
    payload: Dict[str, Optional[Union[str, int, float]]]
) -> Dict[str, Optional[Union[str, int, float]]]:
    """Duplicate all fields with _copy suffix for form filling."""
    duplicated_payload = payload.copy()
    
    # Add duplicated fields with _copy suffix, but only for original fields (not already copied ones)
    for field_id, value in payload.items():
        if not field_id.endswith('_copy'):
            duplicated_payload[f"{field_id}_copy"] = value
    
    return duplicated_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate T4 values via OpenAI using only a personal information."
    )
    parser.add_argument(
        "--personal-info",
        required=True,
        help="Personal information to seed the LLM prompt (e.g., 'John Doe is a software developer at Tech Innovations Inc. and lives at 123 Main St, Anytown, USA').",
    )
    parser.add_argument(
        "--fields",
        type=str,
        required=True,
        help="Path to fields JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the sample t4 values JSON file",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation of generated values",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        field_definitions = load_fields(args.fields)
    except OSError as exc:
        print(f"Failed to read fields file: {exc}", file=sys.stderr)
        return 1

    prompt = build_prompt(personal_info=args.personal_info, fields=field_definitions)

    try:
        generated = request_values(prompt)
    except Exception as exc:
        print(f"OpenAI generation failed: {exc}", file=sys.stderr)
        return 1

    payload = to_full_payload(field_definitions, generated)

    # Validate the generated values (unless skipped)
    if not args.skip_validation:
        validation_result = validate_generated_values(payload, field_definitions)
        
        # Report validation results
        if validation_result.warnings:
            print("Validation warnings:", file=sys.stderr)
            for warning in validation_result.warnings:
                print(f"  - {warning}", file=sys.stderr)
        
        if not validation_result.is_valid:
            print("Validation errors found:", file=sys.stderr)
            for error in validation_result.errors:
                print(f"  - {error}", file=sys.stderr)
            print("Generated values may not be realistic. Consider regenerating.", file=sys.stderr)
        else:
            print("✓ All validation checks passed")

    # Duplicate all fields with _copy suffix for form filling
    duplicated_payload = duplicate_payload(payload)

    with open(args.output, "w") as f:
        json.dump(duplicated_payload, f, indent=4)

    print(f"Wrote generated values to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
