# Form-based Document Generation

## Input Configuration
To generate multiple different type of documents for different users, the configuration files are as follows:
1. (Required) attributes files: Find all fields needed to be filled in for each document (could be done manually or other tools such as OmniParser). There are some samples in the `examples/attributes` directory.
2. (Required) sample images: The sample images of the each document. There are some samples in the `examples/images` directory.
3. (Optional) coordinates files: The coordinates of the fields in the document. There are two types of documents generation: 1. The synthetic document has to follow the structure of the sample document, so the coordinates for each field are required in order to fill in the generated values. 2. The synthetic document can have a different structure, therefore the coordinates are not required. There are some samples in the `examples/coordinates` directory.


## Coordinate Mapping
"coordinate_mapper.py" script is used to map the coordinates of the fields in the document. It is done manually by clicking on the top left corner of the field and then the bottom right corner. For floating point fields, there are two regions to be mapped: the integer part and the decimal part.

Parameters:

- `--image`: Path to the document image (e.g. `data/t4.png`)
- `--fields`: Path to the attributes JSON file (e.g. `examples/attributes/t4.json`)
- `--coordinates`: Path to save the coordinates JSON file (e.g. `examples/coordinates/t4.json`)

Example:
```bash
python coordinate_mapper.py --image examples/images/t4.png --fields examples/attributes/t4.json --coordinates examples/coordinates/t4.json
```

## Run the pipeline
The entire pipeline includes 3 steps: value generation, image generation, and image perturbation.
1. Value Generation
First, it will generate the user profiles. Then for each user profile, it will generate the attribute values for each document based on the given attributes types. Those values will be used as the labels for the document extraction tasks.
2. Image Generation
Based on the generated values and the sample images of each document, generate the synthetic documents for each. If the document has a fixed structure, then the coordinates of the fields will be used to fill in the values. If the document doesn't have a fixed structure, then the LLM will be used to generate the document image based on the values and the sample images.
3. Image Perturbation
Apply multiple perturbation effects (i.e. rotation, lighting, blur, noise) to the synthetic document images from step 2 to generate more diverse images.

Example:
```bash
python main.py --attributes-dir examples/attributes --sample-images-dir examples/images --coordinates-dir examples/coordinates --output-dir ./results --max-workers 4 --num-persona 2
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