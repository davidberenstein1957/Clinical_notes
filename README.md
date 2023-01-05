# Clinical_notes
Training a model to perform Named Entity Recognition on clinical notes

## Task
Extract information from clinical notes.

## Data
(This data has been synthetically generated using GPT-3)

Clinical_note = Jane Smith, 28 [age: 28], was found to have stage II breast cancer [diagnosis: stage II breast cancer]. She underwent surgery to remove the tumor [treatment: surgery] and will also receive radiation therapy [treatment: radiation therapy] and tamoxifen [treatment: tamoxifen]. The surgery went well, and the pathology report showed that the cancer had not spread to the lymph nodes. Jane is understandably anxious about the diagnosis, but her medical team has reassured her that the prognosis is generally good for this type of cancer with appropriate treatment. She has a history of allergies and has had some minor allergic reactions to medications in the past. Jane is a non-smoker and exercises regularly.

Entities = ['age', 'name', 'disease', 'treatment', 'biomarker', 'laterality']

## Discussion

### Step 1. Annotate the data
In order to bootstrap the annotation, we can pre-annotate using spacy.EntityRuler (https://spacy.io/api/entityruler) and  concise-concepts (https://github.com/Pandora-Intelligence/concise-concepts). Then the pre-annotated texts can be uploaded to Argilla (https://www.argilla.io/) for revision and to manually improve the annotation. Once all the clinical notes have been annotated, we have a good starting point to start training the model to perform Named Entity Recognition. 

We can also use keyword extraction models like https://github.com/MaartenGr/KeyBERT and https://huggingface.co/ml6team/keyphrase-extraction-kbir-inspec


### Step 2. Train a model
Now we have to choose a model from the many available. It is advisable to choose a pre-trained model to leverage transfer learning. If the model has been trained with medical data, the transfer of knowledge might be higher than using a model train with law texts. In any case, there are many models out there and we have to choose one. 
Another thing to consider is the embeddings, as there are multiple ways to create the vector representation of the words, and that may impact the performance of the model using a word embedding model that is specific to a particular domain could help. BioWordVec (https://github.com/ncbi-nlp/BioWordVec) I’m not sure when and how to do that. Do all the models work with all the embeddings? Are the embeddings part of the model?
We tried those models:
•	en_core_web_lg (https://spacy.io/models/en)
•	PubMed-w2v.bin (https://bio.nlplab.org/)


### Other resources
- Medspacy (https://github.com/medspacy/medspacy)
