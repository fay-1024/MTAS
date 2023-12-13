## Experiment Replication Package

We provide the codes to replicate our experiment, including the pre-trained model being tested and all the data supporting the experiment.


### 1. Abstractive Summarization Models Under Test

All the codes for the experimental models are stored `model` directory.

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

### 2. Data Source

*All experimental data is stored in `data` directory.*

#### Abstractive Summarization Datasets

In our experiment, we select source inputs from the dev split of XSum and Newsroom datasets.

* `xsum_dev.jsonl`: Detailed information on this dataset can be found at <https://github.com/EdinburghNLP/XSum>.
* `newsroom_validation_data.jsonl`: Detailed information on this dataset can be found at <https://lil.nlp.cornell.edu/newsroom/>.


#### SCY Evaluation Datasets

* `sts.jsonl`: Semantic Textual Similarity (STS) corpus. We use this dataset to evaluate the semantic comprehension of SCY.
* `frank.jsonl`: We use this dataset to evaluate SCY's ability to recognize factual inconsistencies. Detailed information on this dataset can be found at <https://github.com/artidoro/frank>.




































