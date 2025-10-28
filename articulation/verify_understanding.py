# articulation/verify_understanding.py

"""
This script tests that LLM to see how well it understands why it made certain classifications.
We take the same data as we used for running the category classifier, and we present it with a test.
We evaluate how well it's able to describe why it made the classifications it did.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import random
from pandas import read_csv
import csv

load_dotenv()

client = OpenAI()

MCQ_OPTIONS = [
    "The labels are classifying sentences that have less than 40 words.",
    "The labels are classifying sentences that contain an adjective.",
    "The labels are classifying sentences that have periods.",
    "The labels are classifying sentences that relate to history."
]

NONE_OPTION = "None of the options are correct."

language_model = "o4-mini-2025-04-16"

def load_dataset(file_path):
    df = read_csv(file_path, quotechar='"', escapechar='\\')
    return list(zip(df["sentence"], df["is_related"]))

def test_mcq_articulation(topic, data, incl_none_option=False):
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

    base_prompt += "Based on the examples above, answer the following multiple choice question to best describe the nature of the classification of the True labels:\n"

    true_option = NONE_OPTION if incl_none_option else f"The labels are classifying sentences that relate to {topic}."

    shuffled_options = MCQ_OPTIONS.copy()

    if incl_none_option:
        shuffled_options.append(NONE_OPTION)
    else:
        shuffled_options.append(true_option)

    random.shuffle(shuffled_options)

    answer_key = {option: chr(97 + i) for i, option in enumerate(shuffled_options)}
    true_answer = answer_key[true_option]

    mcq_text = "Which of the following best describes the nature of the classification of the True labels?\n"
    mcq_text += "\n".join([f"{v}) {k}" for k, v in answer_key.items()])
    mcq_text += "\nPlease respond with ONLY the letter corresponding to your choice, and nothing else. The valid responses are: a, b, c, d, or e."

    prompt = base_prompt + f"{mcq_text}"

    # Retry up to 3 times if response is not valid
    max_retries = 3
    final_response = None
    
    for attempt in range(max_retries):
        # Use the OpenAI API to get predictions
        response = client.chat.completions.create(
            model=language_model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Parse the response
        for choice in response.choices:
            response_text = choice.message.content.strip().lower()
            
            # Validate that response is either "true" or "false"
            if response_text in ["a", "b", "c", "d", "e"]:
                final_response = response_text
                break
            else:
                print(f"Invalid response on attempt {attempt + 1}: '{response_text}'. Retrying...")
        
        # If we got a valid response, break out of retry loop
        if final_response is not None:
            break
    
    # If after all retries we still don't have a valid response, default to False
    if final_response is None:
        print(f"Failed to get valid response for topic: '{topic}'. Raise an exception.")
        raise ValueError("Could not get valid response from LLM.")
    
    return true_answer == final_response

def test_freeform_articulation(topic, data):
    # Rather than MCQ, we can also do freeform text and see if the LLM mentions the topic to explain the classification rules.
    n_training_samples = 5

    # The first n positive and negative examples are used as few-shot examples. Positive examples are where is_related is True, negative where False.
    few_shot_positives = [s for s, is_rel in data if is_rel][:n_training_samples]
    few_shot_negatives = [s for s, is_rel in data if not is_rel][:n_training_samples]

    base_prompt = "Below, you will find labeled examples that reflect whether the sentences are relevant to our classification or not.\n\n"

    for s in few_shot_positives:
        base_prompt += f"True: {s}\n"
    for s in few_shot_negatives:
        base_prompt += f"False: {s}\n"
    base_prompt += "\n"

    base_prompt += "Based on the examples above, please explain in a few sentences the nature of the classification of the True labels.\n"

    # Use the OpenAI API to get predictions
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": base_prompt}
        ]
    )

    # Parse the response
    for choice in response.choices:
        response_text = choice.message.content.strip().lower()
        if response_text:
            return response_text

    return None


def main():
    # First, get the topic_grades.csv
    grades_df = read_csv(f"data/grades/topic_grades_{language_model.replace('-', '_')}.csv")

    articulation_results = f"data/articulation/articulation_results_{language_model.replace('-', '_')}.csv"

    if not os.path.exists(articulation_results):
        with open(articulation_results, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["category", "understands_mcq", "understands_mcq_none", "freeform_response", "ff_mentions_topic"])

    for index, row in grades_df.iterrows():
        topic = row["category"]
        accuracy = row["accuracy"]
        precision = row["precision"]
        recall = row["recall"]
        f1 = row["f1"]

        print(f"Topic: {topic}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        if accuracy >= 0.9:
            print(f"The LLM demonstrates a strong understanding of the '{topic}' category with an accuracy of {accuracy:.2f}. Continuing with articulation assessment.")
            # Raw data file path
            data_file = f"data/{topic}_dataset.csv"
            data = load_dataset(data_file)
            understands = test_mcq_articulation(topic, data)

            freeform_response = None
            mentions_topic = False
            understands_none = False
            if understands:
                print(f"The LLM has demonstrated an understanding of its classifications for the '{topic}' category.\n")
                freeform_response = test_freeform_articulation(topic, data)
                if freeform_response:
                    print(f"Freeform response for topic '{topic}': {freeform_response}.")
                    print(f"Mentions topic: {topic.lower() in freeform_response.lower()}\n")
                    mentions_topic = topic.lower() in freeform_response.lower()
                understands_none = test_mcq_articulation(topic, data, incl_none_option=True)
            else:
                print(f"The LLM has NOT demonstrated an understanding of its classifications for the '{topic}' category.\n")

            with open(articulation_results, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([topic, understands, understands_none, freeform_response, mentions_topic])
            
        else:
            print(f"Skipping articulation assessment for '{topic}' due to low accuracy of {accuracy:.2f}.\n")

if __name__ == "__main__":
    main()