import json
import jsonlines
import nltk
import numpy
import spacy
from spacy.symbols import nsubj, VERB, AUX

nlp = spacy.load("en_core_web_trf")


def make_first_letter_lower(str):
    return str[:1].lower() + str[1:]


def tree_to_list(node, tree):
    if node.n_lefts + node.n_rights > 0:
        tree.insert(0, node.orth_)
        return [tree_to_list(child, tree) for child in node.children]
    else:
        tree.insert(0, node.orth_)


class MR_adv:

    def transform_sent(self, sentence):
        doc = nlp(sentence)
        tokens = [token for token in doc]
        real_tokens = [token.text.strip() for token in tokens]
        sent_dep = [token.dep_ for token in doc]
        word_index = []
        tree1 = []

        for word in doc:
            child_dep = [child.dep_ for child in list(word.children)]
            child_tag = [child.tag_ for child in list(word.children)]

            if word.dep_ == "ROOT":
                if "nsubj" in child_dep:
                    place_tree1 = child_dep.index("nsubj")
                elif "nsubjpass" in child_dep:
                    place_tree1 = child_dep.index("nsubjpass")
                elif "csubj" in child_dep:
                    place_tree1 = child_dep.index("csubj")
                elif "csubjpass" in child_dep:
                    place_tree1 = child_dep.index("csubjpass")
                else:
                    continue
                tree_to_list(list(word.children)[place_tree1], tree1)
                for leaf in tree1:
                    word_index.append(real_tokens.index(leaf))

        if word_index == []:
            return ""

        legal_pos = numpy.max(word_index)
        word = []
        which = 0
        for token in tokens[legal_pos + 1:]:
            if token.dep_ in ["prep", "mark", "advmod"]:
                word.append(token.text.strip())
                which = real_tokens.index(token.text.strip())

        if word == []:
            return ""

        prep = ["when", "in", "at", "on", "if"]
        first_word = ["am", "is", "are", "was", "were"]
        if str(tokens[0].text).strip() in first_word and " there " not in sentence:
            return ""
        sent = ""
        if word[-1] in prep:
            this_prep = " " + word[-1] + " "
            if this_prep not in sentence:
                return ""
            position = sentence.index(this_prep)
            part_1 = sentence[0:position]
            part_2 = sentence[position + 1:]
            sent = part_2 + ", " + part_1
        return sent


    def generate_new_sent(self, sent):
        new_sent = ""
        start = sent.split(" ")[0]

        if start == "The":
            new_sent = make_first_letter_lower(sent)
        elif start == "I":
            new_sent = sent
        else:
            # 获得专有名词列表
            doc = nlp(sent)
            proper = []
            for ent in doc.ents:
                if ent.label_ != "CARDINAL" and ent.label_ != "ORDINAL" and ent.label_ != "MONEY":
                    proper.append(ent.text)

            if len(proper) == 0:
                new_sent = make_first_letter_lower(sent)

            if len(proper) != 0:
                proper_str = " ".join(proper)
                proper_str_list = proper_str.split(" ")
                if start in proper_str_list:
                    new_sent = sent
                else:
                    new_sent = make_first_letter_lower(sent)

        return new_sent


    def get_follow_doc(self, doc):
        sentences = nltk.tokenize.sent_tokenize(doc)
        change_count = 0
        new_doc = ""
        for j in range(0, len(sentences)):
            sentence = sentences[j]
            new_sentence = ""
            nlp_sent = nlp(sentence)
            verb_list = [token for token in nlp_sent if token.pos == VERB or token.pos == AUX]
            if len(verb_list) > 1 or '"' in sentence or sentence[-1] != ".":
                new_sentence = sentence
            else:
                s = self.transform_sent(sentence[:-1])
                if s != "":
                    sentence = self.generate_new_sent(sentence)
                    new_sentence = self.transform_sent(sentence[:-1]) + "."
                    new_sentence = new_sentence[:1].upper() + new_sentence[1:]
                    change_count += 1
                else:
                    new_sentence = sentence

            if j == len(sentences) - 1:
                new_doc = new_doc + new_sentence
            else:
                new_doc = new_doc + new_sentence + "\n"

        if change_count == 0:
            return ""
        else:
            return new_doc





