# Schema-based Form Filling

To generate a new task form with new values, the pipeline is as follows:
1. Find all fields to be filled in the form (could be done manually or using LLM). The sample is saved in `utils/fields.json`.
2. Run `coordinate_mapper.py` to manually find the coordinates of the fields in the form.
3. Run `value_generator.py` to generate the values for the fields.
4. Run `form_filler.py` to fill the form with the values (step 3) in the correct coordinates (step 2).

## Coordinate Mapping

Coordinate mapping is done manually by clicking on the top left corner of the field and then the bottom right corner. For floating point fields, there are two regions to be mapped: the integer part and the decimal part.

Parameters:

- `--image`: Path to the T4 form image (e.g. `data/t4.png`)
- `--fields`: Path to the fields JSON file (e.g. `utils/fields.json`)
- `--coordinates`: Path to save the coordinates JSON file (e.g. `data/t4_coord.json`)

Example:
```bash
python coordinate_mapper.py --image data/t4.png --fields utils/fields.json --coordinates data/t4_coord.json
```

## Value Generation

Value generation is done by running `value_generator.py` with the following parameters:

- `--job-title`: Job title to seed the LLM prompt (e.g. `Software Developer`)
- `--fields`: Path to the fields JSON file (e.g. `utils/fields.json`)
- `--output`: Path to save the values JSON file (e.g. `data/sample_t4_values.json`)

Example:
```bash
python value_generator.py --job-title "Software Developer" --fields utils/fields.json --output data/sample_t4_values.json
```

## Form Filling

Form filling is done by running `form_filler.py` with the following parameters:

- `--image`: Path to the T4 form image (e.g. `data/t4.png`)
- `--coordinates`: Path to the coordinates JSON file (e.g. `data/t4_coord.json`)
- `--values`: Path to the values JSON file (e.g. `data/sample_t4_values.json`)
- `--output`: Path to save the filled form (e.g. `data/t4_filled.png`)

Example:
```bash
python form_filler.py --image data/t4.png --coordinates data/t4_coord.json --values data/sample_t4_values.json --output data/t4_filled.png
```