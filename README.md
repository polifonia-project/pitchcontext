# pitchcontext
Library for melody analysis based on pitch context vectors.

## Prerequisites:
- lilypond installed and in command line path
- convert (ImageMagick) installed and in command line path
- kernfiles and corresponding melodic features available

## Installation
Use the provided pyproject.toml and poetry. In root of the rep do:
```
poetry install
```
This creates a virtual environment with pitchcontext installed.

## Examples
Requires a Python3 environment with both pitchcontext and streamlit installed.
Two examples are provided:
- apps/st_consonance.py
- apps/st_novelty.py

To run:
```
$ streamlit run st_consonance.py -krnpath <path_to_kern_files> -jsonpath <path_to_json_files>
```
