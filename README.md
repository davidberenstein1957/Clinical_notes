# Clinical_notes
Training a model to perform Named Entity Recognition on clinical notes

## Task
Extract information from clinical notes.

## Data
(This data has been synthetically generated using GPT-3)

Clinical_note = Jane Smith, 28 [age: 28], was found to have stage II breast cancer [diagnosis: stage II breast cancer]. She underwent surgery to remove the tumor [treatment: surgery] and will also receive radiation therapy [treatment: radiation therapy] and tamoxifen [treatment: tamoxifen]. The surgery went well, and the pathology report showed that the cancer had not spread to the lymph nodes. Jane is understandably anxious about the diagnosis, but her medical team has reassured her that the prognosis is generally good for this type of cancer with appropriate treatment. She has a history of allergies and has had some minor allergic reactions to medications in the past. Jane is a non-smoker and exercises regularly.

Entities = ['age', 'name', 'disease', 'treatment', 'biomarker', 'laterality']

## Discussion

### Step 1. Pre-annotate the data
In order to bootstrap the annotation, we can pre-annotate Enities dirrectly using `spacy.EntityRuler` (https://spacy.io/api/entityruler) and  `concise-concepts` (https://github.com/Pandora-Intelligence/concise-concepts). Alernatively, We can also use keyword extraction models like https://github.com/MaartenGr/KeyBERT and https://huggingface.co/ml6team/keyphrase-extraction-kbir-inspec to get pre-annotated keyword. 

Then the pre-annotated texts can be uploaded to Argilla (https://www.argilla.io/) for revision and to manually improve the annotation. Once all the clinical notes have been annotated, we have a good starting point to start training the model to perform Named Entity Recognition. 

#### Step 1.1 Pre-annotate using `spacy.EntityRuler`

The `spacy.EntityRuler` just matches on exact words.
```python
import spacy

data = {
    "bodypart": ["ear", "head", "shoulder"],
    "disease": ["fever", "headache", "cold", "cancer"],
}


nlp = spacy.blank("en")
nlp.add_pipe('sentencizer')
ruler = nlp.add_pipe("entity_ruler")

patterns = []
for key, value in data.items():
    patterns.append({"label": key, "pattern": value})
ruler.add_patterns(patterns)
```

#### Step 1.2 Pre-annotate using `concise-concepts`

`concise-concepts` using word similarity of based on pre-trained word2vec models to enhance matching capabilitief of training data like the ones used in `spacy.EntityRuler`.
It is advisable to choose a pre-trained model to leverage transfer learning. If the model has been trained with medical data, the transfer of knowledge might be higher than using a model train with law texts. In any case, there are many models out there and we have to choose one. 
Another thing to consider is the embeddings, as there are multiple ways to create the vector representation of the words, and that may impact the performance of the model using a word embedding model that is specific to a particular domain could help. BioWordVec (https://github.com/ncbi-nlp/BioWordVec) I’m not sure when and how to do that. Do all the models work with all the embeddings? Are the embeddings part of the model?
We tried those models:
•	en_core_web_lg (https://spacy.io/models/en)
•	PubMed-w2v.bin (https://bio.nlplab.org/)

```python
import spacy
from spacy import displacy

import concise_concepts

# get model from https://bio.nlplab.org/
# model_path = "PubMed-w2v.bin"

data = {
    "body": ["ear", "head", "shoulder"],
    "disease": ["fever", "headache", "cold", "cancer"],
    "treatment": ["surgery"]
    
}

text = """Jane Smith, 28 [age: 28], was found to have stage II breast cancer [diagnosis: stage II breast cancer]. She underwent surgery to remove the tumor [treatment: surgery] and will also receive radiation therapy [treatment: radiation therapy] and tamoxifen [treatment: tamoxifen]. The surgery went well, and the pathology report showed that the cancer had not spread to the lymph nodes. Jane is understandably anxious about the diagnosis, but her medical team has reassured her that the prognosis is generally good for this type of cancer with appropriate treatment. She has a history of allergies and has had some minor allergic reactions to medications in the past. Jane is a non-smoker and exercises regularly."""

nlp = spacy.load("en_core_web_lg", disable=["ner"])
nlp.add_pipe(
    "concise_concepts",
    config={
        "data": data,
        "topn": [20, 20, 20],
        # "model_path": model_path,
        # "include_compound_words": True,
        
    },
)
doc = nlp(text)
[(ent.text, ent.label_) for ent in doc.ents]
```

#### Step 1.3 Pre-annotate using keyword extraction models

Keywords are not alligned with the actual entity labels, so these keywords should then be fine-tuned and corrected within the UI. This means we already see some suggestions of relevant words which we need to fine-tune lateron. Alternatively, the labelled keyword could also be assigned to one of the subgroups, based on most-similar word2vec search to further fine-tune your pre-annotations.

```python
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np

# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])

text = """Jane Smith, 28 [age: 28], was found to have stage II breast cancer [diagnosis: stage II breast cancer]. She underwent surgery to remove the tumor [treatment: surgery] and will also receive radiation therapy [treatment: radiation therapy] and tamoxifen [treatment: tamoxifen]. The surgery went well, and the pathology report showed that the cancer had not spread to the lymph nodes. Jane is understandably anxious about the diagnosis, but her medical team has reassured her that the prognosis is generally good for this type of cancer with appropriate treatment. She has a history of allergies and has had some minor allergic reactions to medications in the past. Jane is a non-smoker and exercises regularly."""

# Load pipeline
model_name = "ml6team/keyphrase-extraction-kbir-inspec"
extractor = KeyphraseExtractionPipeline(model=model_name)

import spacy

nlp = spacy.blank("en")
nlp.add_pipe('sentencizer')
ruler = nlp.add_pipe("entity_ruler")

patterns = []
for word in extractor(text):
    ruler.add_patterns([{"label": "KEYWORD", "pattern": word}])
    
```

#### 1.4 use weak supervision

https://docs.argilla.io/en/latest/tutorials/notebooks/labelling-tokenclassification-skweak-weaksupervision.html#2.-Use-Argilla-to-write-skweak-heuristic-rules

### Step 2. Training a model

After annotating, we can start training our model https://docs.argilla.io/en/latest/guides/features/datasets.html#TokenClassification.

```python
import spacy

nlp = spacy.blank("en")
spacy_bin = dataset_rg.prepare_for_training(framework="spacy", lang=nlp)
```

The spaCy bin can then be used in https://spacy.io/usage/training.

### Other resources
- Medspacy (https://github.com/medspacy/medspacy)
