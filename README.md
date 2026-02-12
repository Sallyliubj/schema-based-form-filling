# Form-based Document Generation

## Input Configuration
Use YAML config file (e.g. `config/forms.yaml`) to define each form and how it should be generated.

Each form entry includes:
1. `form_type`: The form name (e.g. `t4`, `paystub`).
2. `mode`: Generation mode, either:
   - `template`: Fill values into a fixed template image using coordinates.
   - `llm`: Generate a form image using LLM from one or more reference images.
3. `attributes`: Path to the attributes JSON for the form.
4. Mode-specific paths:
   - `template`: requires `image` and `coordinates`.
   - `llm`: requires `images` (list).

Example `config/forms.yaml`:
```yaml
forms:
  - form_type: t4
    mode: coordinate
    attributes: data/forms/t4.json
    template_image: assets/templates/t4.jpg
    coordinates: assets/coords/t4.json

  - form_type: t5
    mode: coordinate
    attributes: data/forms/t5.json
    template_image: assets/templates/t5.jpg
    coordinates: assets/coords/t5.json

  - form_type: paystub
    mode: llm
    attributes: data/forms/paystub.json
    reference_images:
      - assets/reference/paystub_1.jpg
      - assets/reference/paystub_2.jpg

  - form_type: property_tax
    mode: llm
    attributes: data/forms/property_tax.json
    reference_images:
      - assets/reference/property_tax_1.jpg

  - form_type: noa
    mode: llm
    attributes: data/forms/noa.json
    reference_images:
      - assets/reference/noa_1.jpg
```


## Coordinate Mapping
"coordinate_mapper.py" script is used to map the coordinates of the fields in the document. It is done manually by clicking on the top left corner of the field and then the bottom right corner. For floating point fields, there are two regions to be mapped: the integer part and the decimal part.

Parameters:

- `--image`: Path to the document image (e.g. `data/t4.png`)
- `--fields`: Path to the attributes JSON file (e.g. `data/attributes/t4.json`)
- `--coordinates`: Path to save the coordinates JSON file (e.g. `examples/coordinates/t4.json`)

Example:
```bash
python coordinate_mapper.py --image data/images/t4.jpg --fields data/attributes/t4.json --coordinates data/coordinates/t4.json
```

## Run the pipeline
The entire pipeline includes 3 steps: value generation, image generation, and image perturbation.
1. Value Generation
First, it will generate the user profiles. Then for each user profile, it will generate the attribute values for each document based on the given attributes types. Those values will be used as the labels for the document extraction tasks.
2. Image Generation
Based on the generated values and the sample images of each document, generate the synthetic documents for each. If the document has a fixed structure, then the coordinates of the fields will be used to fill in the values. If the document doesn't have a fixed structure, then the LLM will be used to generate the document image based on the values and the sample images.
3. Image Perturbation
Apply multiple perturbation effects (i.e. rotation, lighting, blur, noise) to the synthetic document images from step 2 to generate more diverse images.

Parameters:

- `--config`: Path to YAML config file describing each form (`template` or `llm`) and its paths (required)
- `--output-dir`: Path to the output directory which will contain the generated values and images (required, default: `results`)
- `--max-workers`: Maximum number of parallel workers (optional, default: `4`)
- `--num-persona`: Number of personas to generate (optional, default: `10`)
- `--provider`: Provider to use for generation (optional, default: `openai`, choices: `openai`, `azure`)
- `--text-model`: Model to use for text generation (optional, default: `gpt-5`)
- `--image-model`: Model to use for image generation (optional, default: `gpt-5`)

### Azure OpenAI Configuration
When using `--provider azure`, the following environment variables are required:
- `AZURE_CLIENT_ID`: The client ID of the user-assigned managed identity.
- `AZURE_RESOURCE_NAME`: The name of the Azure OpenAI resource.
- `AZURE_OPENAI_API_VERSION`: The API version to use (e.g. `2024-02-15-preview`).

Note: The `--text-model` and `--image-model` parameters should be the deployment names of the models in your Azure OpenAI resource.


Example:
```bash
python main.py --config config/forms.yaml --output-dir ./results --max-workers 4 --num-persona 2
```

## Result Structure
The sample result structure is as follows:
```
results/
├── values/
    ├── <profile_id>/
        ├── user_profile.json
        ├── t4.json
        ├── t5.json
        ├── property_tax.json
        ├── noa.json
        └── paystub.json
    ...
├── images/
    ├── <profile_id>/
        ├── t4_synthetic.png
        ├── t5_synthetic.png
        ├── property_tax_synthetic.png
        ├── noa_synthetic.png
        └── paystub_synthetic.png
    ...
└── images_perturbed/
```

The `values` directory contains the result for step 1. The `images` directory contains the result for step 2. The `images_perturbed` directory contains the result for step 3.