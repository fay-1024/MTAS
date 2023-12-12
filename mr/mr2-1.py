import nltk
import json
import jsonlines
import torch
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def ent_score(p, h):
    tokenizer = AutoTokenizer.from_pretrained("tomhosking/deberta-v3-base-debiased-nli")
    model = AutoModelForSequenceClassification.from_pretrained("tomhosking/deberta-v3-base-debiased-nli")
    features = tokenizer(p, h, padding=True, truncation=True, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        score = model(**features).logits.softmax(dim=1)[0].tolist()[0]
    return score


class MR2_1:

    def get_most_rel_sen(self, doc, summary):
        sentences = nltk.tokenize.sent_tokenize(doc)
        scores = []
        max_score = 0.0
        max_index = -1
        for j in range(0, len(sentences)):
            sentence = sentences[j]
            predicted_probability = ent_score(sentence, summary)
            scores.append(predicted_probability)
            if predicted_probability > max_score:
                max_score = predicted_probability
                max_index = j
        return sentences[max_index], max_index, scores


    def get_rephrased_sen(self, sen):
        rephrased_sen = ''
        url = "https://paraphrase-genius.p.rapidapi.com/dev/paraphrase/"

        headers = {
            "content-type": "application/json",
            "X-RapidAPI-Key": "...",
            "X-RapidAPI-Host": "..."
        }

        payload = {"text": sen, "result_type": "single"}
        response = requests.post(url, json=payload, headers=headers).json()

        if isinstance(response, list):
            rephrased_sen = response[0]

        return rephrased_sen





















