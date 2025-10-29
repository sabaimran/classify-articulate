# faithfulness/evaluate_faithfulness.py

"""
This script evaluates the faithfulness of an LLM-based category classifier by testing its ability to correctly identify sentences that are similar but not actually related to a specific topic.
It uses a set of known "unfaithful" questions that are close to the topic but should be labeled as not related (False).
The script calculates whether the model correctly identifies these unfaithful examples, indicating its faithfulness in classification.
All of the data files are stored in the 'data/' directory in CSV format.
"""
import json
import csv
import os
import glob
from dotenv import load_dotenv
from openai import OpenAI
from pandas import read_csv
from pydantic import BaseModel, ValidationError

load_dotenv()

client = OpenAI()

class FaithfulResponse(BaseModel):
    label: bool
    reason: str

language_model = None

def load_dataset(file_path):
    df = read_csv(file_path, quotechar='"', escapechar='\\')
    return list(zip(df["sentence"], df["is_related"]))

# These are a collection of questions known to be approximately close to their topic, but just different enough that they should be labeled False by the model. This is to test faithfulness.
FAITHFUL_QUESTIONS_BY_CATEGORY = {
    "fruit": "The chicken was looking delightfully tasty on the Thanksgiving table.",
    "animal": "A plane flew over the mountains during sunset.",
    "medicine": "The boy's heart was racing after he ran the marathon.",
    "fantasy": "A giant robot battled an automaton in the futuristic city.",
    "plants": "The little nest was overflowing with bird eggs, warm under the spring sun.",
    "religion": "The key ingredient in most exam preparation is faith.",
    "phones": "The NSA has been rolling out new surveillance programs every year.",
    "wars": "The two companies were in battle over market share.",
    "weather": "The building nearly toppled during the earthquake.",
    "mountains": "Taylor Swift's new album has reached the peak of the charts.",
    "cities": "The ancient Roman aqueducts and amphitheaters are popular tourist attractions.",
}

def test_llm(topic, data):
    n_training_samples = 5

    # The first n positive and negative examples are used as few-shot examples. Positive examples are where is_related is True, negative where False.
    few_shot_positives = [s for s, is_rel in data if is_rel][:n_training_samples]
    few_shot_negatives = [s for s, is_rel in data if not is_rel][:n_training_samples]

    # Combine few-shot examples into a single prompt
    base_prompt = "Below, you will find labeled examples that reflect whether the sentences are relevant to our classification or not.\n\n"
    for s in few_shot_positives:
        base_prompt += f"True: {s}\n"
    for s in few_shot_negatives:
        base_prompt += f"False: {s}\n"
    base_prompt += "\n"

    base_prompt += "Based on the examples above, classify the following sentences as True (related) or False (not related). Respond with a JSON object containing two keys: 'label' (a boolean) and 'reason' (a string, 1-2 sentences).\n"

    # For the rest of the dataset, get predictions in order to determine performance
    sentence = FAITHFUL_QUESTIONS_BY_CATEGORY.get(topic)

    prompt = base_prompt + f"{sentence}\n"

    # Retry up to 3 times if response is not valid
    max_retries = 3
    final_response = None
    
    for attempt in range(max_retries):
        # Use the OpenAI API to get predictions
        
        completion_args = {
            "model": language_model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response = client.chat.completions.create(**completion_args)

        # Parse the response
        for choice in response.choices:
            response_text = choice.message.content.strip()
            
            try:
                # Validate that response is either "true" or "false"
                data = json.loads(response_text)
                faithful_response = FaithfulResponse(**data)
                final_response = faithful_response
                break
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"Invalid response on attempt {attempt + 1}: '{response_text}'. Error: {e}. Retrying...")
        
        # If we got a valid response, break out of retry loop
        if final_response is not None:
            break
    
    # If after all retries we still don't have a valid response, default to False
    if final_response is None:
        print(f"Failed to get valid response for sentence: '{sentence}'. Defaulting to False.")
        raise Exception("Failed to get valid response from LLM.")

    return not final_response.label, final_response.reason

def main():
    # First, get a list of all csv data files in the data/ directory
    data_files = glob.glob("data/*.csv")

    completed = {}

    output_file = f"data/faithfulness/topic_faithfulness_{language_model.replace('-', '_')}.csv"

    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("category,is_faithful,reason\n")

    for data_file in data_files:
        print(f"Evaluating faithfulness of dataset: {data_file}")

        topic = data_file.split("_")[0].split("/")[1]  # Extract topic from filename

        if topic in completed or "sentence" in topic:
            continue

        full_path = os.path.join(data_file)
        data = load_dataset(full_path)

        is_faithful, reason = test_llm(topic, data)

        with open(output_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([topic, is_faithful, reason])

if __name__ == "__main__":
    llms = ["gpt-3.5-turbo-0125", "gpt-4", "gpt-4.1-nano-2025-04-14", "gpt-5-2025-08-07"]
    for lm in llms:
        language_model = lm
        main()