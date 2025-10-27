from datasets import load_dataset
from random import randint

ENGLISH_DATASET = "agentlans/high-quality-english-sentences"

SENTENCE_COUNT = 200

negative_sentences = []

def main():
    dataset = load_dataset(ENGLISH_DATASET)
    # Given a set of parameters, generate synthetic data for testing purposes.
    print("Generating synthetic data...")

    index = 0
    used_idx = set()
    max_index = len(dataset["train"]) - 1
    while len(negative_sentences) < SENTENCE_COUNT:
        negative_sentences.append(dataset["train"][index]["text"])
        used_idx.add(index)
        index = randint(0, max_index)
        while index in used_idx:
            index = randint(0, max_index)

    with open("base_sentences.txt", "w") as f:
        for sentence in negative_sentences:
            f.write(sentence + "\n")

    print("Synthetic data generation complete.")


if __name__ == "__main__":
    main()
