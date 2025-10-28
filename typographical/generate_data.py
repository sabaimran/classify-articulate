# category/generate_data.py

"""
We have base_sentence.txt, which contains 200 high-quality English sentences, each on a new line. 
We have a particular category that we want to match - in this case, topic-related sentences.
Using this base dataset, we first load the sentences and use them as reference to generate n new sentences related to the topic.
Then, we select n examples of sentences that are NOT related to the topic from the base dataset.
This constructs our final .csv dataset with positive and negative labeled examples, which will be used for few-shot 'training' and evaluation.
"""

import os
from typing import List
from dotenv import load_dotenv
from random import shuffle

load_dotenv()

base_sentence_file = "data/base_sentences.txt"

def load_base_sentences(file_path):
    sentences = []
    with open(file_path, "r") as f:
        sentences = [line.strip() for line in f.readlines()]

    # shuffle the sentences to ensure randomness
    shuffle(sentences)
    return sentences

def get_sentence_length_samples(base_sentences: List[str], target_length:int, target_count=60):
    """Select negative and positive samples from the base sentences.

    Logic:
    - Parse through all base sentences. If the sentence length is less than or equal to target_length, it's a positive sample.
    - If the sentence length is greater than target_length, it's a negative sample.
    - Continue until we have target_count samples for both positive and negative.
    """
    negative_sentences = []
    positive_sentences = []

    for s in base_sentences:
        if len(s.split()) <= target_length:
            positive_sentences.append(s)
        else:
            negative_sentences.append(s)
        if len(negative_sentences) >= target_count and len(positive_sentences) >= target_count:
            break

    return negative_sentences[:target_count], positive_sentences[:target_count]

def get_comma_samples(base_sentences: List[str], target_count=60):
    """
    Select negative and positive samples from the base sentences that contain commas.
    """
    negative_sentences = []
    positive_sentences = []

    for s in base_sentences:
        if "," in s:
            positive_sentences.append(s)
        else:
            negative_sentences.append(s)

        if len(negative_sentences) >= target_count and len(positive_sentences) >= target_count:
            break

    return negative_sentences[:target_count], positive_sentences[:target_count]

def get_decimal_samples(base_sentences: List[str], target_count=60):
    """
    Select negative and positive samples from the base sentences that contain decimal numbers.
    """
    negative_sentences = []
    positive_sentences = []

    for s in base_sentences:
        if any(char.isdigit() for char in s):
            positive_sentences.append(s)
        else:
            negative_sentences.append(s)

        if len(negative_sentences) >= target_count and len(positive_sentences) >= target_count:
            break

    return negative_sentences[:target_count], positive_sentences[:target_count]

def get_character_samples(base_sentences: List[str], target_char_count: str, target_count=60):
    """
    Select negative and positive samples from the base sentences based on character count.
    """
    negative_sentences = []
    positive_sentences = []

    for s in base_sentences:
        if len(s) >= target_char_count:
            positive_sentences.append(s)
        else:
            negative_sentences.append(s)

        if len(negative_sentences) >= target_count and len(positive_sentences) >= target_count:
            break

    return negative_sentences[:target_count], positive_sentences[:target_count]

def get_special_character_samples(base_sentences: List[str], target_count=60):
    """
    Select negative and positive samples from the base sentences based on presence of special characters.
    """
    negative_sentences = []
    positive_sentences = []

    special_chars = set('!@#$%^&*()-_=+[]{}|;:\'"<>?/`~')

    for s in base_sentences:
        if any(char in s for char in special_chars):
            positive_sentences.append(s)
        else:
            negative_sentences.append(s)

        if len(negative_sentences) >= target_count and len(positive_sentences) >= target_count:
            break

    return negative_sentences[:target_count], positive_sentences[:target_count]

def main():
    target_length = 25 # Target sentence length for positive samples
    # Open base sentences file, which is one level up in the parent directory.
    base_sentences_file = os.path.join(os.path.dirname(__file__), f"../{base_sentence_file}")
    base_sentences = load_base_sentences(base_sentences_file)

    negative_sentences, positive_sentences = get_sentence_length_samples(base_sentences, target_length=target_length)

    with open(f"data/sentence_{target_length}_dataset.csv", 'w') as f:
        f.write("sentence,is_related\n")
        for sentence in positive_sentences:
            f.write(f'"{sentence}",True\n')
        for sentence in negative_sentences:
            f.write(f'"{sentence}",False\n')

    negative_sentences, positive_sentences = get_comma_samples(base_sentences)

    with open(f"data/sentence_comma_dataset.csv", 'w') as f:
        f.write("sentence,is_related\n")
        for sentence in positive_sentences:
            f.write(f'"{sentence}",True\n')
        for sentence in negative_sentences:
            f.write(f'"{sentence}",False\n')

    negative_sentences, positive_sentences = get_decimal_samples(base_sentences)

    with open(f"data/sentence_decimal_dataset.csv", 'w') as f:
        f.write("sentence,is_related\n")
        for sentence in positive_sentences:
            f.write(f'"{sentence}",True\n')
        for sentence in negative_sentences:
            f.write(f'"{sentence}",False\n')

    target_char_count = 100
    negative_sentences, positive_sentences = get_character_samples(base_sentences, target_char_count=target_char_count)

    with open(f"data/sentence_{target_char_count}_char_dataset.csv", 'w') as f:
        f.write("sentence,is_related\n")
        for sentence in positive_sentences:
            f.write(f'"{sentence}",True\n')
        for sentence in negative_sentences:
            f.write(f'"{sentence}",False\n')

    negative_sentences, positive_sentences = get_special_character_samples(base_sentences)
    with open(f"data/sentence_special_char_dataset.csv", 'w') as f:
        f.write("sentence,is_related\n")
        for sentence in positive_sentences:
            f.write(f'"{sentence}",True\n')
        for sentence in negative_sentences:
            f.write(f'"{sentence}",False\n')

if __name__ == "__main__":
    main()