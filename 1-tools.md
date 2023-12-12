## Implementation and Instruction of MTAS

We implement MTAS with a set of Python scripts. It is used to generate the follow-up test cases and measure the violations with our proposed new metric SCY.

---

### Abstractive Summarization Models Under Test
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

### Follow-up Test Cases Generation

All the codes for generating follow-up test cases are stored in `mr` directory

* `mr1-1.py`:
* `mr2-1.py`:
* `mr2-2.py`:
* `mr-syn.py`:
* `mr-adv.py`:


























