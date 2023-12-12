import json

import jsonlines
import spacy
from spacy.tokens import Doc
nlp = spacy.load("en_coreference_web_trf")

class MR1_1:
    def resolve_references(self, doc: Doc) -> str:
        cnt = 0
        change_str = ''
        output_string = ''

        not_first_mention_list = ['his', 'her', 'my', 'your', 'its', 'our', 'their', 'I', 'you', 'he', 'she', 'it', 'we',
                                  'they', 'me', 'him', 'us', 'them']
        special_list = ['his', 'her', 'my', 'your', 'its', 'our', 'their']
        # token.idx : token.text
        token_mention_mapper = {}
        output_string = ""
        clusters = [
            val for key, val in doc.spans.items() if key.startswith("coref_cluster")
        ]

        for cluster in clusters:
            first_mention = cluster[0]
            if first_mention.text.lower() in not_first_mention_list:
                continue
            for mention_span in list(cluster)[1:]:
                if ':' in mention_span.text or '"' in mention_span.text:
                    continue
                if mention_span.text.lower() in special_list:
                    token_mention_mapper[mention_span[0].idx] = first_mention.text + "'s" + mention_span[0].whitespace_
                    # print(mention_span.text, '-->', first_mention.text + "'s")
                    cnt += 1
                    change_str += mention_span.text + '-->' + first_mention.text + "'s" + "; "
                else:
                    token_mention_mapper[mention_span[0].idx] = first_mention.text + mention_span[0].whitespace_
                    # print(mention_span.text, '-->', first_mention.text)
                    cnt += 1
                    change_str += mention_span.text + '-->' + first_mention.text + "; "

                for token in mention_span[1:]:
                    token_mention_mapper[token.idx] = ""

        if cnt <= 10:
            for token in doc:
                if token.idx in token_mention_mapper:
                    output_string += token_mention_mapper[token.idx]
                else:
                    output_string += token.text + token.whitespace_

        return output_string, cnt, change_str



if __name__ == '__main__':
    mr = MR1_1()
    text = 'Philip plays the bass because he loves it.'
    doc = nlp(text)
    print(mr.resolve_references(doc))

