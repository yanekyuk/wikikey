import os
from dotenv import load_dotenv
load_dotenv()
# Environment Variables

DATASET = os.getenv('DATASET') or (input('Enter the dataset path (ex. datasets/wikikey-en): ') or None)
if DATASET == None:
  print('No dataset path provided. Exiting...')
  exit()

CHECKPOINT = os.getenv('CHECKPOINT') or (input('Enter the checkpoint for the base model (ex. bert-base-uncased): ') or None)
if CHECKPOINT == None:
  print('No checkpoint provided. Exiting...')
  exit()

TASK_TYPE = os.getenv('TASK_TYPE') or ('extractor' if input('Enter the task type. (e)xtractor/(d)iscriminator): ') == 'e' else 'discriminator')
if TASK_TYPE not in ['extractor', 'discriminator']:
  print('Invalid task type. Exiting...')
  exit()

MODEL_NAME = os.getenv('MODEL_NAME') or (input('Enter the model name (ex. bert-keyword-extractor): ') or None)
if MODEL_NAME == None:
  print('No model name provided. Exiting...')
  exit()

SAMPLE_SIZE = os.getenv('SAMPLE_SIZE') or (int(input(f"Enter the sample size for training (ex. 10000): ")) or 10000)
EPOCHS = int(os.getenv('EPOCHS') or (int(input('Enter the number of epochs (default 8): ')) or 8))
LEARNING_RATE = os.getenv('LEARNING_RATE') or (float(input('Enter the learning rate (ex. 2e-5): ')) or 2e-5)
BATCH_SIZE = int(os.getenv('BATCH_SIZE') or (int(input('Enter the batch size (ex. 16): ')) or 16))

"""## Import packages"""

import wandb, numpy as np, pandas as pd
from datasets import load_metric, load_from_disk
from tabulate import tabulate
from torch import cuda
from transformers import AutoModelForTokenClassification, AutoTokenizer, \
  TrainingArguments, Trainer, DataCollatorForTokenClassification, pipeline

WANDB_API_KEY = os.getenv('WANDB_API_KEY')
WANDB_ENTITY = os.getenv('WANDB_ENTITY')
WANDB_PROJECT = os.getenv('WANDB_PROJECT')
USE_WANDB = WANDB_API_KEY and WANDB_ENTITY and WANDB_PROJECT
if USE_WANDB:
  print('Wandb is enabled')
else:
  print('Wandb is disabled')

# Load the dataset
print('Loading the datasets...')
datasets = load_from_disk(DATASET)
# datasets = load_dataset(DATASET)

# Shuffle the dataset (for random sampling)
print('Shuffling the datasets...')
datasets['train'] = datasets['train'].shuffle(seed=42)
datasets['eval'] = datasets['eval'].shuffle(seed=42)
datasets['test'] = datasets['test'].shuffle(seed=42)

# Filter the dataset by removing the sentences with no concepts tagged.
print('Filtering the datasets...')
datasets['train'] = datasets['train'].filter(lambda x: len(x['concepts']) > 0)
datasets['test'] = datasets['test'].filter(lambda x: len(x['concepts']) > 0)
datasets['eval'] = datasets['eval'].filter(lambda x: len(x['concepts']) > 0)

# Get a smaller sample from the dataset in order compare the performance of the model between languages.
print('Getting a smaller sample from the dataset...')
datasets['train'] = datasets['train'].select(range(SAMPLE_SIZE))
datasets['test'] = datasets['test'].select(range(int(SAMPLE_SIZE / 10)))
datasets['eval'] = datasets['eval'].select(range(int(SAMPLE_SIZE / 10)))

PUBLISH_TO_HUGGINGFACE = os.getenv('PUBLISH_TO_HUGGINGFACE') or (True if input('Publish to Huggingface? (y/n): ') == 'y' else False)
# Set the dataset labels and label maps.
print('Setting the dataset labels...')
id2label, label2id = {}, {}
# Extractor BIO-Tagging
if TASK_TYPE == 'extractor':
  label_names = ["O", "B-KEY", "I-KEY"]
  id2label, label2id = {}, {}
  for key, value in enumerate(label_names):
    id2label[key] = value
    label2id[value] = key
  del key, value
# Discriminator BIO-Tagging
if TASK_TYPE == 'discriminator':
  label_names = ["O", "B-ENT", "I-ENT", "B-CON", "I-CON"]
  for key, value in enumerate(label_names):
    id2label[key] = value
    label2id[value] = key
  del key, value

# Tokenizer
# Fetch the tokenizer from the base model.
print('Fetching the tokenizer from the base model...')
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

def match_list_to_labels(tokenized_sample, tokenized_keys, key_type: str):
  """Create labels for the tokenized sample based on the tokenized keywords."""
  b_key, i_key = 1, 2
  if TASK_TYPE == 'discriminator':
    b_key = 1 if key_type == 'entity' else 3
    i_key = 2 if key_type == 'entity' else 4
  
  for row in range(len(tokenized_keys)):
    keys = [list(filter(lambda x: x not in tokenizer.all_special_ids, key_list)) for key_list in tokenized_keys[row]]
    sentence_input_ids = tokenized_sample['input_ids'][row]
    for key in keys:
      key_len = len(key)
      for i in range(0, len(sentence_input_ids) - key_len):
        if sentence_input_ids[i] in tokenizer.all_special_ids:
          tokenized_sample['labels'][row][i] = -100
        if key == sentence_input_ids[i : i + key_len]:
          for j in range(i, i + key_len):
            tokenized_sample['labels'][row][j] = b_key if i == j else i_key
  return tokenized_sample['labels']

def tokenize (samples):
  """Tokenize the samples."""
  tokenized_sample = tokenizer.batch_encode_plus(samples['sentence'], truncation=True, max_length=512)
  tokenized_sample['labels'] = [np.zeros(len(input_ids), dtype=int).tolist() for input_ids in tokenized_sample.input_ids]
  entity_lists = [entity_list for entity_list in samples['entities']]
  concept_lists = [concept_list for concept_list in samples['concepts']]
  # Extractor strategy
  if TASK_TYPE == 'extractor':
    keyword_lists = []
    for i in range(len(entity_lists)):
      keyword_list = sorted(list(set(entity_lists[i]).union(set(concept_lists[i]))), key=len)
      keyword_lists.append(keyword_list)
    tokenized_keywords = [tokenizer.batch_encode_plus(keyword_list, add_special_tokens=False)['input_ids'] if len(keyword_list) > 0 else [] for keyword_list in keyword_lists]
    tokenized_sample['labels'] = match_list_to_labels(tokenized_sample, tokenized_keywords, 'keyword')
  # Discriminator strategy
  if TASK_TYPE == 'discriminator':
    tokenized_entities = [tokenizer.batch_encode_plus(entity_list, add_special_tokens=False)['input_ids'] if len(entity_list) > 0 else [] for entity_list in entity_lists]
    tokenized_concepts = [tokenizer.batch_encode_plus(concept_list, add_special_tokens=False)['input_ids'] if len(concept_list) > 0 else [] for concept_list in concept_lists]
    tokenized_sample['labels'] = match_list_to_labels(tokenized_sample, tokenized_concepts, 'concept')
    tokenized_sample['labels'] = match_list_to_labels(tokenized_sample, tokenized_entities, 'entity')
  # Return results
  return tokenized_sample

print("Tokenizing the datasets...")
tokenized_datasets = datasets.map(tokenize, batched=True)

# Check whether it works as intended.
def printAsTable (index, count):
  sample_dataset = tokenized_datasets['train'].select(range(index + count))
  for i in range(index, index + count):
    labels = sample_dataset[i]['labels']
    input_ids = sample_dataset[i]['input_ids']
    rows = [labels, sample_dataset[i]['entities'], sample_dataset[i]['concepts']]
    headers = [tokenizer.decode(input_id) for input_id in input_ids]
    table = tabulate(rows, headers=headers)
    print("Example result from tokenization:\n", table + "\n")

printAsTable(1, 5)

# Clear the unnecessary columns in the dataset.
print('Clearing the unnecessary columns in the dataset...')
tokenized_datasets = tokenized_datasets.remove_columns(['sentence', 'entities', 'concepts', 'page_key'])

# Load the model from the checkpoint
print('Loading the model from the checkpoint...')
model = AutoModelForTokenClassification.from_pretrained(
  CHECKPOINT,
  num_labels=len(label_names),
  id2label=id2label,
  label2id=label2id,
)
print("Model configuration: \n ", model.config)

# Define the metric computation method
print("Defining the metric computation method...")
metric = load_metric("seqeval")

def compute_metrics(p):
  predictions, labels = p
  predictions = np.argmax(predictions, axis=2)
  # Remove ignored index (special tokens)
  true_predictions = [
      [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
      for prediction, label in zip(predictions, labels)
  ]
  true_labels = [
      [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
      for prediction, label in zip(predictions, labels)
  ]
  
  results = metric.compute(predictions=true_predictions, references=true_labels)
  print("results: \n ", results)
  if TASK_TYPE == 'discriminator':
    return {
      "precision": results["overall_precision"],
      "recall": results["overall_recall"],
      "accuracy": results["overall_accuracy"],
      "f1": results["overall_f1"],

      "ent/precision": results["ENT"]["precision"],
      "ent/accuracy": results["ENT"]["recall"], 
      "ent/f1": results["ENT"]["f1"],

      "con/precision": results["CON"]["precision"],
      "con/accuracy": results["CON"]["recall"], 
      "con/f1": results["CON"]["f1"],
    }
  if TASK_TYPE == 'extractor':
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "accuracy": results["overall_accuracy"],
        "f1": results["overall_f1"]
    }

# Setup the data collator
# - We are doing the padding during the training
print("Setting up the data collator...")
data_collator = DataCollatorForTokenClassification(
    tokenizer,
    padding=True,
)

# Set the training preferences.
default_args = {
  "output_dir": MODEL_NAME,
  "evaluation_strategy": "epoch",
  "save_strategy": "epoch",
  "push_to_hub": True,
  "load_best_model_at_end": True,
  "metric_for_best_model": "f1",
  "report_to": "wandb" if USE_WANDB else "console",
}

# Set the training arguments.
training_args = TrainingArguments(
  **default_args,
  # Learning args
  learning_rate=2e-5,
  num_train_epochs=8,
  weight_decay=0.01,
  # Device args
  per_device_train_batch_size=16,
  per_device_eval_batch_size=16,
  # gradient_checkpointing=True,
  fp16=True if cuda.is_available() else False,
)

# Create the trainer.
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['eval'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start training the model.
trainer.train()

# Send the report to Weights & Biases and model to Hugging Face
if USE_WANDB:
  print('Saving the report to Weights & Biases...')
  wandb.finish()
if PUBLISH_TO_HUGGINGFACE:
  print('Saving the model to Hugging Face...')
  trainer.push_to_hub()

# Try it out!
trained_model = pipeline(
  task="token-classification",
  model=f"{MODEL_NAME}",
  aggregation_strategy="simple"
)

sentence = input('Enter a sentence: ')

print(trained_model(sentence))

