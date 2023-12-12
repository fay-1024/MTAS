import json
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import jsonlines

class Models:

    def get_bart_summary(self, document):
        bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")
        bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-xsum")
        ARTICLE_TO_SUMMARIZE = document
        inputs = bart_tokenizer(ARTICLE_TO_SUMMARIZE, max_length=1024, truncation=False, return_tensors="pt")
        summary_ids = bart_model.generate(inputs["input_ids"])
        return bart_tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


    def get_pegasus_summary(self, document):
        pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
        pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
        ARTICLE_TO_SUMMARIZE = document
        inputs = pegasus_tokenizer(ARTICLE_TO_SUMMARIZE, max_length=1024, truncation=False, return_tensors="pt")
        summary_ids = pegasus_model.generate(inputs["input_ids"])
        return pegasus_tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def get_t5_summary(self, text):
        t5_tokenizer = AutoTokenizer.from_pretrained("sysresearch101/t5-large-finetuned-xsum")
        t5_model = AutoModelForSeq2SeqLM.from_pretrained("sysresearch101/t5-large-finetuned-xsum")
        input_ids = t5_tokenizer.encode(text, max_length=1024, truncation=False, return_tensors="pt")
        summary_ids = t5_model.generate(input_ids, min_length=20, max_length=80, num_beams=10, repetition_penalty=2.5, length_penalty=1.0, early_stopping=True, no_repeat_ngram_size=2, use_cache=True, do_sample=True, temperature=0.8, top_k=50, top_p=0.95)
        summary_text = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary_text


