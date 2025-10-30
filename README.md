# Classify - Articulate

This research projects defines a test environment in which we generate classification examples for a model and then test its ability to:
1. Correctly identify labels given a set of few-shot examples
2. Articulate why it is correctly able to label examples
3. Adhere to the stated reasoning that it has presented

You can find the [final report here](https://docs.google.com/document/d/1LYOWdXPI-d9tsiGkZvIMtW8hYXgEDCcc1QZ_EyA8dgY/edit?usp=sharing).

## Setup

This project uses `uv` for package management. To get started, follow these steps:

1.  Install `uv`:

    ```bash
    pip install uv
    ```

2.  Create a virtual environment:

    ```bash
    uv venv
    ```

3.  Install the dependencies:

    ```bash
    uv pip sync pyproject.toml
    ```

## Code Structure

Here is a brief overview of the Python scripts in this project:

*   `get_sentences.py`: Pulls high quality english sentences from `agentlans/high-quality-english-sentences` on HF to set the base data for out dataset generation.
*   `grade_classifier.py`: Evaluates the performance of an LLM-based category classifier by grading its predictions against a labeled dataset.
*   `articulation/verify_understanding.py`: Tests how well the LLM understands why it made certain classifications.
*   `category/generate_data.py`: Generates topic-related and non-topic-related sentences for training and evaluation.
*   `faithfulness/evaluate_faithfulness.py`: Evaluates the faithfulness of the LLM-based category classifier.
*   `results/make_charts.py`: Generates charts to visualize the performance metrics of the classifier.
*   `typographical/generate_data.py`: Generates datasets based on typographical features like sentence length, commas, decimals, and special characters.
