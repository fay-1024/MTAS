import json
import requests
import jsonlines
import nltk
from nltk.tokenize import word_tokenize

url = "https://twinword-word-associations-v1.p.rapidapi.com/associations/"

headers = {"X-RapidAPI-Key": "...", "X-RapidAPI-Host": "..."}


class MR_syn:
    def get_synonyms(self, word):
        syn_word = ""
        querystring = {"entry": word}
        response = requests.get(url, headers=headers, params=querystring).json()
        if "associations_scored" in response.keys():
            associations_scored = response["associations_scored"]
            syn = max(associations_scored, key=lambda x: associations_scored[x])
            syn_score = associations_scored[syn]
            if syn_score > 0.85:
                syn_word = syn

        return syn_word



















