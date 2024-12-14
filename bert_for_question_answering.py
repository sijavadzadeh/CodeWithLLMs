import torch
from transformers import (
    BertForQuestionAnswering,
    BertTokenizerFast,
)
from scipy.special import softmax
import plotly.express as px
import pandas as pd
import numpy as np

context = "The giraffe is a large African hoofed mammal belonging to the genus Giraffa. It is the tallest living terrestrial animal and the largest ruminant on Earth. Traditionally, giraffes were thought to be one species, Giraffa camelopardalis, with nine subspecies. Most recently, researchers proposed dividing them into up to eight extant species due to new research into their mitochondrial and nuclear DNA, as well as morphological measurements. Seven other extinct species of Giraffa are known from the fossil record."
question = "How many giraffe species are there?"

model_name = "deepset/bert-base-cased-squad2"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)


inputs = tokenizer(question, context, return_tensors="pt")
tokenizer.tokenize(context)

def predict_answer(context, question):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    start_scores, end_scores = softmax(outputs.start_logits)[0], softmax(outputs.end_logits)[0]
    start_idx = np.argmax(start_scores)
    end_idx = np.argmax(end_scores)
    confidence_score = (start_scores[start_idx] + end_scores[end_idx]) /2
    answer_ids = inputs.input_ids[0][start_idx: end_idx + 1]
    answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids)
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    if answer != tokenizer.cls_token:
        return answer, confidence_score
    return None, confidence_score

"""
### Evaluating the Function

Test the `predict_answer` function with various questions and contexts."""

# Defining a new context and predicting answers to some questions
context = """Coffee is a beverage brewed from roasted coffee beans. Darkly colored, bitter, and slightly acidic, coffee has a stimulating effect on humans, primarily due to its caffeine content. It has the highest sales in the world market for hot drinks.[2]

The seeds of the Coffea plant's fruits are separated to produce unroasted green coffee beans. The beans are roasted and then ground into fine particles typically steeped in hot water before being filtered out, producing a cup of coffee. It is usually served hot, although chilled or iced coffee is common. Coffee can be prepared and presented in a variety of ways (e.g., espresso, French press, caffè latte, or already-brewed canned coffee). Sugar, sugar substitutes, milk, and cream are often added to mask the bitter taste or enhance the flavor.

Though coffee is now a global commodity, it has a long history tied closely to food traditions around the Red Sea. The earliest credible evidence of coffee drinking as the modern beverage appears in modern-day Yemen in southern Arabia in the middle of the 15th century in Sufi shrines, where coffee seeds were first roasted and brewed in a manner similar to how it is now prepared for drinking.[3] The coffee beans were procured by the Yemenis from the Ethiopian Highlands via coastal Somali intermediaries, and cultivated in Yemen. By the 16th century, the drink had reached the rest of the Middle East and North Africa, later spreading to Europe.

The two most commonly grown coffee bean types are C. arabica and C. robusta.[4] Coffee plants are cultivated in over 70 countries, primarily in the equatorial regions of the Americas, Southeast Asia, the Indian subcontinent, and Africa. As of 2023, Brazil was the leading grower of coffee beans, producing 35% of the world's total. Green, unroasted coffee is traded as an agricultural commodity. Despite sales of coffee reaching billions of dollars worldwide, farmers producing coffee beans disproportionately live in poverty. Critics of the coffee industry have also pointed to its negative impact on the environment and the clearing of land for coffee-growing and water use. The global coffee industry is massive and worth $495.50 billion as of 2023.[5] Brazil, Vietnam, and Colombia are the top exporters of coffee beans as of 2023. \n Rapid growth in coffee production in South America during the second half of the 19th century was matched by an increase in consumption in developed countries, though nowhere has this growth been as pronounced as in the United States, where a high rate of population growth was compounded by doubling of per capita consumption between 1860 and 1920. Though the United States was not the heaviest coffee-drinking nation at the time (Belgium, the Netherlands and Nordic countries all had comparable or higher levels of per capita consumption), due to its sheer size, it was already the largest consumer of coffee in the world by 1860, and, by 1920, around half of all coffee produced worldwide was consumed in the US.[37]

Coffee has become a vital cash crop for many developing countries. Over one hundred million people in developing countries have become dependent on coffee as their primary source of income. It has become the primary export and economic backbone for African countries like Uganda, Burundi, Rwanda, and Ethiopia,[40] as well as many Central American countries.

"""

len(tokenizer.tokenize(context))
predict_answer(context, "What is coffee?")
predict_answer(context, "What are the most common coffee beans?")
predict_answer(context, "How can I make ice coffee?")
predict_answer(context[4000:], "How many people are dependent on coffee for their income?")

# Defining a function to chunk sentences
def chunk_sentences(sentences, chunk_size, stride):
    chunks = []
    num_sentences = len(sentences)
    for i in range(0, num_sentences, chunk_size - stride):
        chunk = sentences[i: i + chunk_size]
        chunks.append(chunk)
    return chunks

sentences = [
    "Sentence 1.",
    "Sentence 2.",
    "Sentence 3.",
    "Sentence 4.",
    "Sentence 5.",
    "Sentence 6.",
    "Sentence 7.",
    "Sentence 8.",
    "Sentence 9.",
    "Sentence 10."
]


sentences = context.split("\n")
chunked_sentences = chunk_sentences(sentences, chunk_size=3, stride=1)


questions = ["What is coffee?", "What are the most common coffee beans?", "How can I make ice coffee?", "How many people are dependent on coffee for their income?"]

answers = {}

for chunk in chunked_sentences:
    sub_context = "\n".join(chunk)
    for question in questions:
        answer, score = predict_answer(sub_context, question)

        if answer:
            if question not in answers:
                answers[question] = (answer, score)
            else:
                if score > answers[question][1]:
                    answers[question] = (answer, score)

print(answers)

"""## Conclusion and Further Exploration 🌈

Congratulations! You've learned how to utilize BERT for question answering. Experiment with different contexts and questions to explore the model's capabilities further.
"""
