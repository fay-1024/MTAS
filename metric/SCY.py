import json
import nltk
import spacy
from nltk.corpus import stopwords
from sense2vec import Sense2Vec
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("merge_entities")
nlp.add_pipe("merge_noun_chunks")
s2v = Sense2Vec().from_disk("D:\\Datasets\\sense2vec\\s2v_reddit_2015_md\\s2v_old")

tokenizer = AutoTokenizer.from_pretrained("tomhosking/deberta-v3-base-debiased-nli")
model = AutoModelForSequenceClassification.from_pretrained("tomhosking/deberta-v3-base-debiased-nli")


def merge_token_with_label(text, text_list):
    text_doc = nlp(text)
    list_with_label = []
    for token in text_list:
        token_list = token.text.split(' ')
        if len(token_list) == 1:
            if token.pos_ == "PROPN":
                word_string = token.text + "|NOUN"
            else:
                word_string = token.text + "|" + token.pos_
        else:
            word_string = "_".join(token_list).lower() + "|" + token.pos_
        list_with_label.append(word_string)
    return list_with_label


def sim1(w1, w1_text, token):
    nlp.remove_pipe("merge_entities")
    nlp.remove_pipe("merge_noun_chunks")

    token_doc = nlp(token)
    # token_list = [w for w in token_doc if not w.text.lower() in stopwords.words('english')]
    token_list = [w for w in token_doc]
    # print(token_list)
    if len(token_list) < 1:
        similar = 0
    else:
        scores = []
        for t in token_list:
            s = t.text + "|" + t.pos_
            if s in s2v:
                si = s2v.similarity(w1, s)
            else:
                si = int(w1_text.lower() == t.text.lower())
            scores.append(si)
        similar = sum(scores)/len(scores)
        # similar = max(scores)

    nlp.add_pipe("merge_entities")
    nlp.add_pipe("merge_noun_chunks")
    return similar


def sim2(text_1, text_2):
    nlp.remove_pipe("merge_entities")
    nlp.remove_pipe("merge_noun_chunks")

    s1_doc = nlp(text_1)
    s2_doc = nlp(text_2)
    # s1_list = [w for w in s1_doc if not w.text.lower() in stopwords.words('english')]
    # s2_list = [w for w in s2_doc if not w.text.lower() in stopwords.words('english')]
    s1_list = [w for w in s1_doc]
    s2_list = [w for w in s2_doc]

    if len(s1_list) < 1 or len(s2_list) < 1:
        similar = 0
    else:
        scores = []
        for t1 in s1_list:
            scores_list = []
            s1 = t1.text + "|" + t1.pos_
            if s1 in s2v:
                for t2 in s2_list:
                    s2 = t2.text + "|" + t2.pos_
                    if s2 in s2v:
                        si = s2v.similarity(s1, s2)
                    else:
                        si = int(t1.text.lower() == t2.text.lower())
                    scores_list.append(si)
            else:
                for t2 in s2_list:
                    si = int(t1.text.lower() == t2.text.lower())
                    scores_list.append(si)
            scores.append(max(scores_list))
        similar = sum(scores) / len(scores)
        # similar = max(scores)
    nlp.add_pipe("merge_entities")
    nlp.add_pipe("merge_noun_chunks")
    return similar


def cos_sim(ref_text, sys_text):
    source_doc = nlp(ref_text)
    generated_doc = nlp(sys_text)
    ref_list = [token for token in source_doc if (token.text != "" and token.pos_ != "PUNCT")]
    sys_list = [token for token in generated_doc if (token.text != "" and token.pos_ != "PUNCT")]
    # print(ref_list)
    # print(sys_list)
    # 删除停用词
    ref_list = [w for w in ref_list if not w.text.lower() in stopwords.words('english')]
    sys_list = [w for w in sys_list if not w.text.lower() in stopwords.words('english')]
    # print(ref_list)
    # print(sys_list)

    if len(ref_list) == 0 or len(sys_list) == 0:
        return 0

    if set(ref_list) <= set(sys_list) or set(sys_list) <= set(ref_list):
        return 1

    ref_list_with_label = merge_token_with_label(ref_text, ref_list)
    sys_list_with_label = merge_token_with_label(sys_text, sys_list)
    # print(ref_list_with_label)
    # print(sys_list_with_label)

    scores = []
    for i in range(0, len(sys_list_with_label)):
        word1 = sys_list_with_label[i]
        if word1 in s2v:
            a_list_score = []
            for j in range(0, len(ref_list_with_label)):
                word2 = ref_list_with_label[j]
                if word2 in s2v:
                    similarity = s2v.similarity(word1, word2)

                    # print(word1, "<--->", word2, "==", similarity)

                else:
                    similarity = sim1(word1, sys_list[i].text, ref_list[j].text)

                    # print(word1, "<--->", word2, "==", similarity)

                a_list_score.append(similarity)
            scores.append(max(a_list_score))
        else:
            a_list_score = []
            for j in range(0, len(ref_list_with_label)):
                word2 = ref_list_with_label[j]
                if word2 in s2v:
                    similarity = sim1(word2, ref_list[j].text, sys_list[i].text)
                else:
                    similarity = sim2(sys_list[i].text, ref_list[j].text)

                # print(word1, "<--->", word2, "==", similarity)

                a_list_score.append(similarity)
            scores.append(max(a_list_score))

    score_sum = 0
    weight_sum = 0
    for i in range(0, len(scores)):
        score = scores[i]
        text_list = sys_list[i].text.split(' ')
        # text_list = [w for w in text_list if not w.lower() in stopwords.words('english')]
        weight = len(text_list)
        score_sum = score * weight + score_sum
        weight_sum = weight_sum + weight

    final_sim = score_sum / weight_sum
    return final_sim
    # return sum(scores) / len(scores)


class SCY:
    def get_scy_score(self, summary, new_summary):
        scy = -1
        f1 = 0.75
        f2 = 0.25
        cos_recall = cos_sim(new_summary, summary)

        list_sentence1 = summary.split(" ")
        list_sentence2 = new_summary.split(" ")

        if len(list_sentence1) > len(list_sentence2):
            premise = summary
            hypothesis = new_summary
        else:
            premise = new_summary
            hypothesis = summary

        features = tokenizer(premise, hypothesis, padding=True, truncation=True, return_tensors="pt")
        model.eval()
        with torch.no_grad():
            scores = model(**features).logits
            result = scores.argmax(dim=1).item()
            predicted_probability = scores.softmax(dim=1)[0].tolist()[result]

        if result == 2:
            scy = f1 * cos_recall - f2 * predicted_probability
        elif result == 1:
            scy = f1 * cos_recall
        elif result == 0:
            scy = f1 * cos_recall + f2 * predicted_probability

        return scy














