# grade_classifier.py

"""
This script evaluates the performance of an LLM-based category classifier by grading its predictions against a labeled dataset.
It calculates accuracy, precision, recall, and F1 score to assess how well the model identifies sentences related to a specific topic.
All of the data files are stored in the 'data/' directory in CSV format.
They have two columns: 'sentence' and 'is_related' (True/False). There are 60 positive and 60 negative examples in each file.
"""
import os
import glob
from dotenv import load_dotenv
from openai import OpenAI
from pandas import read_csv

load_dotenv()

client = OpenAI()

language_model = "gpt-4"

def load_dataset(file_path):
    df = read_csv(file_path, quotechar='"', escapechar='\\')
    return list(zip(df["sentence"], df["is_related"]))

def grade_classifier(predictions, ground_truth):
    tp = sum(1 for p, gt in zip(predictions, ground_truth) if p and gt)
    tn = sum(1 for p, gt in zip(predictions, ground_truth) if not p and not gt)
    fp = sum(1 for p, gt in zip(predictions, ground_truth) if p and not gt)
    fn = sum(1 for p, gt in zip(predictions, ground_truth) if not p and gt)

    accuracy = (tp + tn) / len(ground_truth)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1

def test_llm(data):
    n_training_samples = 5

    # The first n positive and negative examples are used as few-shot examples. Positive examples are where is_related is True, negative where False.
    few_shot_positives = [s for s, is_rel in data if is_rel][:n_training_samples]
    few_shot_negatives = [s for s, is_rel in data if not is_rel][:n_training_samples]

    # Retrieve the examples that will be used in the prompt as 'training' for the model
    few_shot_set = set(few_shot_positives + few_shot_negatives)
    rest_of_data = [(s, is_rel) for s, is_rel in data if s not in few_shot_set]

    # Gather the rest of the data that will be used for evaluation
    sentences_to_classify = {s: label for s, label in rest_of_data}

    # Combine few-shot examples into a single prompt
    base_prompt = "Below, you will find labeled examples that reflect whether the sentences are relevant to our classification or not.\n\n"
    for s in few_shot_positives:
        base_prompt += f"True: {s}\n"
    for s in few_shot_negatives:
        base_prompt += f"False: {s}\n"
    base_prompt += "\n"

    base_prompt += "Based on the examples above, classify the following sentences as True (related) or False (not related). Return ONLY 'true' or 'false' and nothing else:\n"

    # For the rest of the dataset, get predictions in order to determine performance
    predictions = {}

    for sentence, _ in sentences_to_classify.items():
        prompt = base_prompt + f"{sentence}\n"

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
                if response_text in ["true", "false"]:
                    final_response = response_text == "true"
                    break
                else:
                    print(f"Invalid response on attempt {attempt + 1}: '{response_text}'. Retrying...")
            
            # If we got a valid response, break out of retry loop
            if final_response is not None:
                break
        
        # If after all retries we still don't have a valid response, default to False
        if final_response is None:
            print(f"Failed to get valid response for sentence: '{sentence}'. Defaulting to False.")
            final_response = False
        
        predictions[sentence] = final_response

    return sentences_to_classify, predictions

def main():
    # First, get a list of all csv data files in the data/ directory
    data_files = glob.glob("data/*.csv")

    completed = {"animal", "cities", "fantasy", "fruit", "wars", "plants", "religion", "weather", "mountains", "phones", "medicine"}

    output_file = f"data/grades/topic_grades_{language_model.replace('-', '_')}.csv"

    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("category,accuracy,precision,recall,f1\n")

    for data_file in data_files:
        print(f"Evaluating dataset: {data_file}")

        topic = data_file.split("_")[0].split("/")[1]  # Extract topic from filename

        if topic == "sentence":
            topic += "_" + data_file.split("_")[1]  # Handle multi-word topics

        if topic in completed:
            continue

        full_path = os.path.join(data_file)
        data = load_dataset(full_path)

        ground_truth_dict, predictions_dict = test_llm(data)

        ground_truth = [ground_truth_dict[s] for s in predictions_dict.keys()]
        predictions = [predictions_dict[s] for s in predictions_dict.keys()]

        accuracy, precision, recall, f1 = grade_classifier(predictions, ground_truth)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")

        with open(output_file, "a") as f:
            f.write(f"{topic},{accuracy:.2f},{precision:.2f},{recall:.2f},{f1:.2f}\n")

if __name__ == "__main__":
    main()