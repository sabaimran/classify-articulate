# Can Models Accurately Explain Their Reasoning?

## Abstract

As AI systems are deployed in higher risk domains, developers set up monitors to ensure adherence to some specified guideline. A base assumption in many monitoring systems is that a model's stated reasoning can reveal its true thought process, allowing the developer to intervene in case of misalignment. However, state reasoning traces are not always reflective of a model's actual thought processes. Models can present plausible, but incorrect or outright dishonest explanations. The goal of this project was to better understand whether models an articulate their reasoning in a classification problem to assess faithfulness. In this experiment, models were presented with a classification task to identify the topic of a sentence. The models were able to almost perfectly identify correct categories and perfectly articulate their reasoning in this project. Further experimentation would be required to administer more difficult tasks in order to elicit greater variance in response adherence.

## Methods

I tested `gpt-3.5-turbo`, `gpt-4`, `gpt-4.1-nano`, and `gpt-5` for completeness across various capability levels.

### Topic-Based Classification 

I wanted to present the model with a classification task related to identifying the topics in a sentence. Typographical assessments (e.g., 'identify all sentences with 200 characters') frequently encounter quality issues due to tokenization gaps, which I wanted to avoid for the scope of the experiment. Topic identification also easily generalizes, making it trivial to extend the experiments in our research apparatus.

The topics chosen were: plants, medicine, phones, religion, wars, fantasy, cities, weather, fruit, animal, and mountains. I chose to select a wide variety of topics to ensure some generalization in the task, while retaining consistency in the type of classification for accurate measurement.

I constructed the classification dataset with a mix of public data and LLM-generated data. To ensure that the sentences used for this classification task reflected real-life variance in human (English) language, I pulled base sentences from this [agentlans - high quality English sentences dataset](https://huggingface.co/datasets/agentlans/high-quality-english-sentences). I did this by writing a script that generates similar data for any topic with a few modifications.

Given these base sentences, for each category, I had an LLM generate sentences that are similar in tone, quality, and length, but limited in content to the specific category being tested (e.g., medicine, plants, etc.). From there, I had high quality, labeled datasets with positive and negative sentences belonging to each of the 11 categories.

To run the classifier on the model, I needed to expose it to labeled sentences for each category. The label reflects whether the sentence belongs in the category (True) or not (False). Given a few examples in its system prompt, the model was then instructed to label an unseen sentence at inference time. I compiled the results across the dataset to assess the model's accuracy, precision, and recall.

For classification tasks where accuracy exceeded 90%, I ran an articulation test. This consisted of four phases:
1. Select the correct classification rule from an MCQ per category.
2. Select 'None of the above' from an MCQ where none of the options are a match.
3. Use a freeform response to describe the classification rule.
4. Use a freeform response to describe the classification rule, being provided a misleading hint.

For part 3, I verified that the topic name or a synonym was present in the freeform response. This was trivial to do with a mix of keyword matching and manual investigation due to the low volume.

### Typographical Challenge

Building on top of the same `agentlas` dataset, I also created a secondary dataset which contained 5 typographical challenges. The 5 different challenges reflect these rules:
- <= 25 words
- \>= 1 commas
- \>= 1 number
- \>= 1 special character (e.g., !#$%)
- \>= 100 characters

## Results

### Topic-Based Classification

Across all categories, models were able to classify appropriately and articulate their reasoning correctly. I tested the four target models across the range of classification tasks, taking into account the accuracy to measure performance.

Most of the models breezed through the topic-based classification tasks, scoring upwards of 95%.

Articulation tests also showed strong model performance. Most of them were able to successfully select the correct topic categories. I increased the difficulty of the MCQ portion by adding other options that were similar but incorrect, which did result in some performance degradation.

Even when reducing the number of samples provided in the few-shot training example (from 5 to 2), I still found that the models were able to accurately describe the categories of the task.

Models demonstrated high performance in the free form articulation task, strongly indicating that the difficulty of the MCQ exam directly caused the poor results, rather than an outright comprehension failure.

I decided to add an additional experimentation layer by providing false hints in the freeform articulation test, and the models (to my surprise) were still able to bypass the red herring and accurately describe the topic category. Under the malicious hint condition, I did notice a slight decrease in the acuity of the free response. Without the bad hint, models had answers that were more distinctly correct, while with the hint, their articulation took a hit on some categories.

Surprisingly, articulation suffered in the condition where the models were presented with the `None of the above` option. This was always the correct option in those instances. This may indicate that, in absence of a clearly correct answer, there is some impulse for models to mark a best approximate as true. This is even when they were able to give good answers to the freeform response.

Because the models had a high accuracy (meaning their hit rates were high both on labeling positive and negative samples), we can be sure that the accurate articulations are representative of their thought processes; when confronted with negative samples, they are accurately able to identify it as failing the classification criteria.

### Typographical Challenge

The models were not able to pass the typographical challenges, so I did not do further examination on the articulation for this suite. On average, they scored between 50-60% on these tasks. It's important to note that I was not able to complete a run with `gpt-5` on this task due to time constraints and long inference times.

## Further Work

The classification task I focused on for the scope of this assessment seemed to be trivially easy for models even from 2-3 years ago. To further assess articulation capabilities, it would be necessary to run a classification task that is slightly more difficult for models to complete.

This may include construction of a dataset with more subtle differences in categorization (e.g., sentences all contain mentions of cities, but only want to select cities > 2M population, sentences all contain a number, but only want to select prime numbers).

It's important to note that tasks that are too difficult will reflect an increase in chain of thought adherence, making it harder to find potentially mismatched reasoning traces.

Where the categorization itself is more difficult, articulation may also become trickier, as models may not be able to pinpoint exactly why they have answered the way they have. To further extend the experiment, it may also be interesting to assess adherence when the topic might be 'taboo' per the training constitution of the model (e.g., related to CBRNs or adult content).

My original intuition was that letter-counting or typographical tests would be less deterministic to evaluate. To verify this, I added in the typographical experiment in which the model was tasked with classifying on conditions such as character count, presence of special characters, and sentence length. These did not meet the accuracy criteria for articulation testing. I think these tests may be good candidates for this experiment if classification can be improved, as the model may not always be able to pinpoint the nature of the classification when it does not elicit a semantic correlation in its activations.

## Code

All relevant code for this project can be found in [sabaimran/classify-articulate](https://github.com/sabaimran/classify-articulate) on GitHub.

The `data` directory contains the datasets used for evaluations. The `data/grades` directory contains raw results from the classifier evaluation, and `data/articulation` has raw results from the articulation tests. 

## References

- [1] [Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/pdf/2307.13702)
- [2] [Language Models Don’t Always Say What They Think](https://arxiv.org/pdf/2305.04388)
- [3] [When Chain of Thought is Necessary, Language Models Struggle to Evade Monitors](https://arxiv.org/pdf/2507.05246)