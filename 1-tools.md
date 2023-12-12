## Implementation and Instruction of MTAS

We implement MTAS with a set of Python scripts. It is used to generate the follow-up test cases and measure the violations with our proposed new metric SCY.

---

### 1. Abstractive Summarization Models Under Test
Our experiments employ three state-of-the-art pre-trained language generation models that have been fine-tuned for the task of abstractive text summarization: BART, Pegasus and T5. All the models are released with detailed configurations on Hugging Face.
* BART: <https://huggingface.co/facebook/bart-large-xsum>
* Pegasus: <https://huggingface.co/google/pegasus-xsum>
* T5: <https://huggingface.co/sysresearch101/t5-large-finetuned-xsum>

The following is an example of running the BART model to obtain a summary:

```python
import json
from model.get_summary import Models

dataset = []
file_path = "xsum_validation_data.jsonl"
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        dataset.append(json.loads(line))
model = Models()
for i in range(0, len(dataset)):
    document = dataset[i]["doc"]
    summary = model.get_bart_summary(document)
    print(summary)
```

### 2. Follow-up Test Cases Generation

The codes for generating follow-up test cases are stored in `mr` directory.

**Main Executable Components:**
* `mr1-1.py`: MR<sub>1-1</sub> constructs follow-up inputs by coreference resolution. Its inputs are source documents.
* `mr2-1.py`: MR<sub>2-1</sub> constructs follow-up inputs by emphasizing key sentence. Its inputs are source documents and summaries.
* `mr2-2.py`: MR<sub>2-2</sub> constructs follow-up inputs by restructuring key sentence. Its inputs are source documents and summaries.
* `mr-syn.py`: MR<sub>w-syn</sub> is designed for word-level perturbations. its inputs are source documents.
* `mr-adv.py`: MR<sub>s-adv</sub> is designed for sentence-level perturbations. its inputs are source documents.

**To automate the generation of follow-up inputs, we employed several text processing tools:**
* `NeuralCoref`, a coreference resolution tool: <https://github.com/huggingface/neuralcoref>
* `Paraphrase Genius API`, a sentence paraphrasing tool:  <https://rapidapi.com/genius-tools-genius-tools-default/api/paraphrase-genius>
* `Spacy`, an industrial-strength natural language processing toolkit:  <https://spacy.io/>
* `Word Associations API`, a tool for obtaining synonyms:  <https://rapidapi.com/twinword/api/word-associations>
























