
import json
from jsonlines import jsonlines
import spacy

nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("merge_entities")
nlp.add_pipe("merge_noun_chunks")


class MR2_2:

    # M1 连词连接两句话，各自有自己的主语
    # The motorcycle rider was a 49-year-old man and the car driver was a 61-year-old man
    def conj_with2subj(self, sen, conj):
        new_sen = ''
        split_mark = " " + str(conj) + " "
        split_list = sen.split(split_mark)  # 根据连词进行切片
        if len(split_list) == 2:
            sen_1 = split_list[0]
            if sen_1.endswith(','):
                sen_1 = sen_1.replace(sen_1[-1], '')
            # new_sen = sen_1 + ". \n" + str(conj)[:1].upper() + str(conj)[1:] + " " + split_list[1]
            new_sen = split_list[1][0].upper() + split_list[1][1:] + "\n" + str(conj)[:1].upper() + str(conj)[
                                                                                                    1:] + " " + \
                      sen_1[0].lower() + sen_1[0][1:] + "."

        return new_sen

    # M2 连词连接两句话，有共同的主语
    # He scored twice for the U's but was unable to save them from relegation.
    def conj_with1subj(self, root, sen, conj):
        new_sen = ''
        subj = [token for token in root.lefts if
                token.dep_ == 'nsubj' or token.dep_ == "nsubjpass" or token.dep_ == "csubj" or token.dep_ == "csubjpass"][
            0]
        split_mark = " " + str(conj) + " "
        split_list = sen.split(split_mark)  # 根据连词进行切片
        if len(split_list) == 2:
            sen_1 = split_list[0]
            if sen_1.endswith(','):
                sen_1 = sen_1.replace(sen_1[-1], '')
            new_sen = sen_1 + ". \n" + str(conj)[:1].upper() + str(conj)[1:] + " " \
                      + str(subj)[0].lower() + str(subj)[1:] + " " + split_list[1]
        return new_sen

    # 调用M1或者M2
    def conj_with2sen(self, sen, root):
        new_sen = ""
        # 取连词
        conj = [token for token in root.rights if token.dep_ == 'cc'][0]
        # 取连词后面的动词
        conj_verb = [token for token in root.rights if token.dep_ == 'conj'][0]
        # 判断连词后面是否还有主语
        has_subj = [token for token in conj_verb.lefts if
                    token.dep_ == 'nsubj' or token.dep_ == "nsubjpass" or token.dep_ == "csubj" or token.dep_ == "csubjpass"]
        # 有主语直接切割
        if len(has_subj) == 1:
            new_sen = self.conj_with2subj(sen, conj)
        elif len(has_subj) == 0:
            new_sen = self.conj_with1subj(root, sen, conj)

        return new_sen

    # M3 处理带which从句的
    # A tow truck had to be called to remove the car which was stuck on the track.
    def relcl_with_which(self, sen, last_verb):
        new_sen = ''
        actual_sub = last_verb.head.text[0].lower() + last_verb.head.text[1:]
        if actual_sub.startswith("a "):
            actual_sub = actual_sub.replace("a ", "the ")
        split_list = sen.split(" which ")  # 根据which进行切片
        if len(split_list) == 2:
            sen_1 = split_list[0]
            if sen_1.endswith(','):
                sen_1 = sen_1.replace(sen_1[-1], '')
            new_sen = sen_1 + ". \n" + "And " + actual_sub + " " + split_list[1]
        return new_sen

    # M4 处理带when从句的
    def relcl_with_when(self, sen):
        new_sen = ''
        split_list = sen.split(" when ")  # 根据which进行切片
        if len(split_list) == 2:
            sen_1 = split_list[0]
            if sen_1.endswith(','):
                sen_1 = sen_1.replace(sen_1[-1], '')
            # new_sen = sen_1 + ". \n" + "At that time, " + split_list[1]
            new_sen = sen_1 + ". \n" + "This was a time when " + split_list[1]
        return new_sen


    def get_splited_sens(self, sen):
        new_sentence = ''
        nlp_doc = nlp(sen)
        # 根节点
        root = [token for token in nlp_doc if token.head == token][0]
        # 获取根节点左右孩子依赖标签
        root_left_dep = [token.dep_ for token in root.lefts]
        root_right_dep = [token.dep_ for token in root.rights]
        # 列表内存元组， 元组有dep标签和头词性标签组成
        # word_tuple = [(token.dep_, token.head.pos_) for token in nlp_doc]
        # 这句话里面的所有动词
        all_verb = [token for token in nlp_doc if token.pos_ == 'VERB']
        len_all_verb = len(all_verb)

        if ('nsubj' in root_left_dep or 'nsubjpass' in root_left_dep or 'csubj' in root_left_dep
                or 'csubjpass' in root_left_dep) and 'cc' in root_right_dep and 'conj' in root_right_dep:
            new_sentence = self.conj_with2sen(sen, root)

        elif len_all_verb != 0 and all_verb[len_all_verb - 1].dep_ == 'relcl':
            last_verb = all_verb[len_all_verb - 1]
            # 主要针对which的情况
            last_verb_has_sub = [token for token in last_verb.lefts if token.dep_ == 'nsubj'
                                     or token.dep_ == "nsubjpass" or token.dep_ == "csubj" or token.dep_ == "csubjpass"]
            # 主要针对 when 的情况
            has_adv = [token for token in last_verb.lefts if token.dep_ == 'advmod']

            if len(last_verb_has_sub) == 1 and last_verb_has_sub[0].text == 'which':
                new_sentence = self.relcl_with_which(sen, last_verb)
            elif len(has_adv) == 1 and has_adv[0].text == 'when':
                new_sentence = self.relcl_with_when(sen)

        return new_sentence






















