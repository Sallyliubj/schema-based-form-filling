# Image Generation Pipeline for Synthetic Documents

This pipeline generates synthetic Canadian tax and financial documents in two stages:
1. **Value Generation**: Creates realistic form values progressively
2. **Image Generation**: Produces filled form images from the generated values

## Workflow

```
Base Profile → Paystub → T4 → T5 → Property Tax → NOA
     ↓            ↓       ↓     ↓         ↓          ↓
  Values      Values   Values Values   Values    Values
     ↓            ↓       ↓     ↓         ↓          ↓
  Images      Images   Images Images   Images    Images
```

The pipeline follows a sequential order where each form builds upon the user profile and previously generated data. Forms are conditionally generated based on user attributes.

## Setup

**Required Form Template Images:**

Before running the image generation pipeline, you must upload the following blank form template images to the `images/` directory:

- `noa.jpg` - Notice of Assessment template
- `paystub.jpg` - Paystub template
- `property_tax.jpg` - Property Tax template
- `t4.jpg` - T4 form template
- `t5.jpg` - T5 form template

These images serve as templates for generating filled forms. The pipeline expects these files to exist in `image_generation_pipeline/images/`.

## Main Functions

### `value_generator.py`

Generates realistic form values using OpenAI API.

**Main Functions:**

- `generate_user_profile() -> UserProfile`
  - Generates a base user profile with demographic and financial attributes
  - Returns: Structured UserProfile object

- `generate_form_values(form_type: str, user_profile: UserProfile, previous_data: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]`
  - Generates form values for a specific form type
  - Parameters:
    - `form_type`: Type of form ('paystub', 't4', 't5', 'property_tax', 'noa')
    - `user_profile`: The base user profile
    - `previous_data`: Dictionary of previously generated form data (optional)
  - Returns: Dictionary of field_id -> value mappings

- `value_generation_pipeline(output_dir: Path = RESULTS_DIR) -> Dict[str, Any]`
  - Runs the full document generation pipeline
  - Parameters:
    - `output_dir`: Directory to save individual form value files (defaults to `results/values/`)
  - Returns: Dictionary containing user_profile and all generated form data


### `image_generator.py`

Generates filled form images from generated values using two methods:
- **Coordinate-based filling** (T4, T5): Uses pre-defined coordinate mappings
- **LLM-based generation** (Paystub, Property Tax, NOA): Uses OpenAI image generation

**Main Functions:**

- `generate_form_image(form_type: str) -> Optional[str]`
  - Generates a filled form image for a specific form type
  - Parameters:
    - `form_type`: Type of form to generate
  - Returns: Path to generated image, or None if failed

- `generate_form_with_llm(form_type: str, source_image_path: str, values: Dict[str, Any], output_path: str, model: str = DEFAULT_MODEL) -> Optional[str]`
  - Generates a filled form image using LLM
  - Parameters:
    - `form_type`: Type of form ('paystub', 'property_tax', 'noa')
    - `source_image_path`: Path to the blank form template
    - `values`: Dictionary of field values to fill
    - `output_path`: Path to save the generated image
    - `model`: OpenAI model to use (defaults to 'gpt-5')
  - Returns: Path to generated image, or None if failed

- `image_generation_pipeline() -> Dict[str, Optional[str]]`
  - Runs the image generation pipeline for all available forms
  - Returns: Dictionary mapping form_type to output path (or None if failed)


## Example

```bash
# Generate form values
python value_generator.py

# Generate filled form images
python image_generator.py
```

## Directory Structure

```
image_generation_pipeline/
├── attributes/          # Form field definitions (JSON)
├── coordinates/         # Coordinate mappings for T4/T5 (JSON)
├── images/             # Form templates (JPG) - user must provide
├── results/
│   ├── values/         # Generated form values (JSON)
│   └── images/         # Generated filled form images (PNG)
└── ...
```