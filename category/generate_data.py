# category/generate_data.py

"""
We have base_sentence.txt, which contains 200 high-quality English sentences, each on a new line. 
We have a particular category that we want to match - in this case, topic-related sentences.
Using this base dataset, we first load the sentences and use them as reference to generate n new sentences related to the topic.
Then, we select n examples of sentences that are NOT related to the topic from the base dataset.
This constructs our final .csv dataset with positive and negative labeled examples, which will be used for few-shot 'training' and evaluation.
"""

from openai import OpenAI
import os
from dotenv import load_dotenv
from random import shuffle

load_dotenv()

base_sentence_file = "data/base_sentences.txt"

TOPIC_OPTIONS = ["fruit", "animal", "medicine", "fantasy", "weather", "phones", "wars", "cities", "mountains", "plants", "religion"]

TOPIC = None

TOPIC_INSTRUCTIONS = {
    "fruit": {
        "generate_instructions": "You are a helpful assistant that generates fruit-related sentence. For instance, the sentences can be about apples, a recipe for banana bread, a child playing with strawberries. It should not be about non-fruit topics. Do not use the word 'fruit' in the sentences. Instead, name specific fruits (e.g., apple, banana, strawberry, orange, grape, mango, etc.). Be creative, use fruits that aren't as common in the lexicon. Use a variation in sentence structure, tone, and style - some should be narrative, others should be scientific, some conversational, some descriptive.",
        "identify_instructions": "Identify which sentences are NOT related to fruit (e.g., not about apples, bananas, berries, citrus, fruit recipes, or fruit descriptions)."
    },
    "animal": {
        "generate_instructions": "You are a helpful assistant that generates animal-related sentences. For instance, the sentences can be about dogs, a story about cats, or facts about elephants. It should not be about non-animal topics. Do not use the word 'animal' in the sentences. Instead, name specific animals (e.g., dog, cat, elephant, lion, tiger, giraffe, etc.). Be creative, use animals that aren't as common in the lexicon. Use a variation in sentence structure, tone, and style - some should be narrative, others should be scientific, some conversational, some descriptive.",
        "identify_instructions": "Identify which sentences are NOT related to animal (e.g., not about dogs, cats, elephants, lions, tigers, giraffes, animal stories, or animal facts)."
    },
    "medicine": {
        "generate_instructions": "You are a helpful assistant that generates medicine-related sentences. For instance, the sentences can be about medications, treatments, or health conditions. It should not be about non-medicine topics. Do not use the word 'medicine' in the sentences. Instead, name specific medications or treatments (e.g., aspirin, chemotherapy, physical therapy, etc.). Be creative, use a variation in sentence structure, tone, and style - some should be narrative, others should be scientific, some conversational, some descriptive.",
        "identify_instructions": "Identify which sentences are NOT related to medicine (e.g., not about medications, treatments, health conditions, or medical procedures)."
    },
    "fantasy": {
        "generate_instructions": "You are a helpful assistant that generates fantasy-related sentences. For instance, the sentences can be about mythical creatures, magical worlds, or epic quests. It should not be about non-fantasy topics. Do not use the word 'fantasy' in the sentences. Instead, name specific elements of fantasy (e.g., dragons, wizards, enchanted forests, etc.). Be creative, use a variation in sentence structure, tone, and style - some should be narrative, others should be descriptive.",
        "identify_instructions": "Identify which sentences are NOT related to fantasy (e.g., not about mythical creatures, magical worlds, or epic quests)."
    },
    "weather": {
        "generate_instructions": "You are a helpful assistant that generates weather-related sentences. For instance, the sentences can be about storms, sunshine, or climate phenomena. It should not be about non-weather topics. Do not use the word 'weather' in the sentences. Instead, name specific weather conditions (e.g., rain, snow, hurricanes, etc.). Be creative, use a variation in sentence structure, tone, and style - some should be narrative, others should be scientific.",
        "identify_instructions": "Identify which sentences are NOT related to weather (e.g., not about storms, sunshine, or climate phenomena)."
    },
    "phones": {
        "generate_instructions": "You are a helpful assistant that generates phone-related sentences. For instance, the sentences can be about smartphones, calls, or mobile technology. It should not be about non-phone topics. Do not use the word 'phone' in the sentences. Instead, name specific phone features or brands (e.g., iPhone, Android, touchscreen, etc.). Be creative, use a variation in sentence structure, tone, and style - some should be narrative, others should be technical.",
        "identify_instructions": "Identify which sentences are NOT related to phones (e.g., not about smartphones, calls, or mobile technology)."
    },
    "wars": {
        "generate_instructions": "You are a helpful assistant that generates war-related sentences. For instance, the sentences can be about battles, military strategies, or historical conflicts. It should not be about non-war topics. Do not use the word 'war' in the sentences, unless it's to name a specific incident. Instead, name specific wars or military terms (e.g., World War II, tactics, soldiers, etc.). Be creative, use a variation in sentence structure, tone, and style - some should be narrative, others should be historical.",
        "identify_instructions": "Identify which sentences are NOT related to wars (e.g., not about battles, military strategies, or historical conflicts)."
    },
    "cities": {
        "generate_instructions": "You are a helpful assistant that generates city-related sentences. For instance, the sentences can be about urban life, landmarks, or city culture. It should not be about non-city topics. Do not use the word 'city' in the sentences. Instead, name specific cities or urban features (e.g., skyscrapers, public transport, nightlife, etc.). Be creative, use a variation in sentence structure, tone, and style - some should be narrative, others descriptive.",
        "identify_instructions": "Identify which sentences are NOT related to cities (e.g., not about urban life, landmarks, or city culture)."
    },
    "mountains": {
        "generate_instructions": "You are a helpful assistant that generates mountain-related sentences. For instance, the sentences can be about hiking, mountain ranges, or alpine environments. It should not be about non-mountain topics. Do not use the word 'mountain' in the sentences. Instead, name specific mountains or related features (e.g., Everest, trails, basecamp, snow-capped peaks, etc.). Be creative, use a variation in sentence structure, tone, and style - some should be narrative, others descriptive.",
        "identify_instructions": "Identify which sentences are NOT related to mountains (e.g., not about hiking, mountain ranges, or alpine environments)."
    },
    "plants": {
        "generate_instructions": "You are a helpful assistant that generates plant-related sentences. For instance, the sentences can be about gardening, types of plants, or botanical facts. It should not be about non-plant topics. Do not use the word 'plant' in the sentences. Instead, name specific plants or gardening terms (e.g., roses, strelitzia, photosynthesis, succulents, etc.). Be creative, use a variation in sentence structure, tone, and style - some should be narrative, others scientific.",
        "identify_instructions": "Identify which sentences are NOT related to plants (e.g., not about gardening, types of plants, or botanical facts)."
    },
    "religion": {
        "generate_instructions": "You are a helpful assistant that generates religion-related sentences. For instance, the sentences can be about religious practices, beliefs, or historical religious events. It should not be about non-religion topics. Do not use the word 'religion' in the sentences. Instead, name specific religions or related terms (e.g., Islam, meditation, rituals, etc.). Be creative, use a variation in sentence structure, tone, and style - some should be narrative, others descriptive.",
        "identify_instructions": "Identify which sentences are NOT related to religion (e.g., not about religious practices, beliefs, or historical religious events)."
    }
}

def load_base_sentences(file_path):
    sentences = []
    with open(file_path, "r") as f:
        sentences = [line.strip() for line in f.readlines()]

    # shuffle the sentences to ensure randomness
    shuffle(sentences)
    return sentences

def generate_topic_sentences(base_sentences, target_count=60):
    """Generate topic-related sentences in batches of 20 per API call.

    Keeps overall behavior the same, but batches requests to reduce round-trips.
    """
    client = OpenAI()

    topic_sentences = []
    batch_size = 20

    # Keep calling until we hit target_count, requesting up to 20 at a time
    while len(topic_sentences) < target_count:
        remaining = target_count - len(topic_sentences)
        n_choices = min(batch_size, remaining)

        prompt = (
            f"Generate {TOPIC}-related sentences, where each sentence is separated by a new line. Use the following examples for reference to demonstrate quality, tone, style, variation, topic, and length of each example output:\n\n"
            + "\n".join(base_sentences[:20])  # Provide a few examples for context
            + f"\n\n{TOPIC}-related sentences:"
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": TOPIC_INSTRUCTIONS[TOPIC]["generate_instructions"]},
                {"role": "user", "content": prompt}
            ],
            max_tokens=8000,
            n=n_choices,
            stop=None,
            temperature=0.8,
        )

        # Collect one or more lines from each choice; trim and filter empties
        for choice in response.choices:
            content = choice.message.content or ""
            for sentence in content.split("\n"):
                s = sentence.strip()
                if s:
                    topic_sentences.append(s)
                    if len(topic_sentences) >= target_count:
                        break
            if len(topic_sentences) >= target_count:
                break

    return topic_sentences[:target_count]

def generate_non_topic_sentences(base_sentences, target_count=60):
    """Select non-topic-related sentences from the base dataset in batches of 20.

    Logic:
    - Present the LLM with 20 base sentences at a time, indexed 0..N-1 for that batch.
    - Ask it to return ONLY the indices of sentences that are NOT related to the topic.
    - Collect those sentences, avoiding duplicates, until we have at least `target_count`.
    - Return the first `target_count` collected.
    """
    client = OpenAI()

    def parse_indices(text: str, max_local_index: int) -> list[int]:
        """Extract zero-based indices from model output safely.

        Accepts formats like: "0,2,5" or lines with numbers. Filters to [0, max_local_index).
        """
        import re

        nums = re.findall(r"\d+", text)
        seen = set()
        out = []
        for n in nums:
            i = int(n)
            if 0 <= i < max_local_index and i not in seen:
                seen.add(i)
                out.append(i)
        return out

    non_topic_sentences: list[str] = []
    selected_global_indices: set[int] = set()
    batch_size = 20

    start = 0
    made_progress_in_pass = True
    # We'll iterate across the dataset in batches of 20; if we reach the end without enough,
    # try one more pass only if we made progress previously, to avoid infinite loops.
    while len(non_topic_sentences) < target_count and (start < len(base_sentences) or made_progress_in_pass):
        if start >= len(base_sentences):
            # New pass
            start = 0
            made_progress_in_pass = False

        batch = base_sentences[start:start + batch_size]
        if not batch:
            break

        # Create a numbered list for the model with local indices
        indexed_lines = [f"{i}: {s}" for i, s in enumerate(batch)]
        instruction = (
            "You will be given a small list of sentences labeled with zero-based indices 0..N-1.\n"
            f"{TOPIC_INSTRUCTIONS[TOPIC]['identify_instructions']}\n"
            "Return ONLY the zero-based indices of the target sentences as a comma-separated list with no spaces (e.g., 0,3,7).\n"
            "Do not include any explanation or extra text.\n"
        )

        user_content = (
            instruction
            + "\nHere are the sentences:\n"
            + "\n".join(indexed_lines)
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"You are a careful classifier that selects non-{TOPIC}-related sentences from a provided list. Output only indices as requested."},
                {"role": "user", "content": user_content},
            ],
            max_tokens=50,
            n=1,
            temperature=0,
        )

        content = response.choices[0].message.content or ""
        local_indices = parse_indices(content, len(batch))

        added_this_batch = 0
        for li in local_indices:
            gi = start + li
            if gi not in selected_global_indices:
                selected_global_indices.add(gi)
                non_topic_sentences.append(base_sentences[gi])
                added_this_batch += 1
                if len(non_topic_sentences) >= target_count:
                    break

        if added_this_batch > 0:
            made_progress_in_pass = True

        start += batch_size

    return non_topic_sentences[:target_count]

def main():
    # Open base sentences file, which is one level up in the parent directory.
    base_sentences_file = os.path.join(os.path.dirname(__file__), f"../{base_sentence_file}")
    base_sentences = load_base_sentences(base_sentences_file)

    topic_sentences = generate_topic_sentences(base_sentences, target_count=50)
    non_topic_sentences = generate_non_topic_sentences(base_sentences, target_count=50)

    with open(f"data/{TOPIC}_dataset.csv", 'w') as f:
        f.write("sentence,is_related\n")
        for sentence in topic_sentences:
            f.write(f'"{sentence}",True\n')
        for sentence in non_topic_sentences:
            f.write(f'"{sentence}",False\n')

    print("Generated topic-related and non-topic-related sentences.")

if __name__ == "__main__":
    for topic in TOPIC_OPTIONS:
        TOPIC = topic
        main()