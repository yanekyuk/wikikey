# Import all the required packages
import json, pandas as pd, requests, re, nltk, numpy as np
import os
from datetime import date, timedelta
from bs4 import BeautifulSoup
from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import notebook_login
from ratelimit import limits, sleep_and_retry
from os.path import exists


# Get environment variables
from dotenv import load_dotenv
load_dotenv()


# Download punkt for sentence tokenization
nltk.download('punkt')


# Set the variables
WIKI_ACCESS_TOKEN = os.getenv('WIKI_ACCESS_TOKEN') or (input('Enter Wikipedia access token (optional): ') or None)
LANGUAGE = os.getenv('LANGUAGE') or (input('Enter language (en, fr, de, es, it, ja, pt, ru, zh): ') or 'en')
USE_HUGGINGFACE = os.getenv('USE_HUGGINGFACE') or (True if input('Do you want to use Huggingface hub? (y/n): ') == 'y' else False)
CONVERT_NUMBERIC_TO_ENTITY = os.getenv('CONVERT_NUMERIC_TO_ENTITY') or True if input('Convert numeric keywords to entity? (y/n): ') else False
# Set the headers values for API requests to Wikipedia
HEADERS = None
if WIKI_ACCESS_TOKEN:
  HEADERS = {
      'Authorization': f'Bearer {WIKI_ACCESS_TOKEN}',
  }


# Test API calls
response = requests.get(
  "https://api.wikimedia.org/core/v1/wikipedia/en/page/Earth/bare",
  headers=HEADERS,
)
print("Testing Wikipedia API resulted in: ", response.status_code)
if response.status_code != 200:
  raise Exception("Wikipedia API call failed.")


def get_featured_page(date: str) -> set[str]:
  """Returns the set of most read page titles from Wikipedia for a given date."""
  # Fetch data by date and convert it to JSON
  url = f'https://api.wikimedia.org/feed/v1/wikipedia/{LANGUAGE}/featured/{date}'
  response = requests.get(url, headers=HEADERS)
  data = response.json()
  
  # Return an empty set if the request does not specify a most read list of pages
  if 'mostread' not in data.keys():
    return set()
  
  articles = data['mostread']['articles']
  # Eliminate the duplicated by using set
  titles = set(list(map(lambda x: x['title'], articles)))
  return titles

def get_featured_pages(count: int = 100, skip: int = 1) -> set[str]:
  """Iterates over the given count of featured pages and returns the set of most read page titles."""
  print(f'Fetching page titles...')
  pages = set()
  end_date = date.today() - timedelta(days=skip)
  for i in tqdm(range(count)):
    end_date -= timedelta(days=1)
    date_str = end_date.strftime('%Y/%m/%d')
    titles = get_featured_page(date_str)
    pages = pages.union(titles)
  return pages


# File persistency for the dataset
def load_from_disk() -> set[str]:
  """Loads the set of most read page titles from disk."""
  pages = []
  with open(f'data/pages.{LANGUAGE}.json') as json_file:
    pages = json.load(json_file)
  return pages

def save_to_disk(pages: list[str]) -> None:
  """Saves the given list of pages to disk."""
  with open(f'data/pages.{LANGUAGE}.json', 'w') as outfile:
    json.dump(pages, outfile)

def load_or_recreate_pages(count: int = 200, extend: bool = False) -> list[str]:
  pages = set()
  file_exists = exists(f'data/pages.{LANGUAGE}.json')
  if file_exists:
    print('Loading file from storage...')
    pages = load_from_disk()
    if extend:
      new_pages = get_featured_pages(count=count, skip=len(pages))
      pages = pages.union(new_pages)
      save_to_disk(list(pages))
  else:
    print('Recreating the file...')
    pages = get_featured_pages(count)
    save_to_disk(list(pages))
  print('Pages are ready.')
  return list(pages)

# Start the page loader
pages = load_or_recreate_pages(count=200, extend=False)

# Rate limit setup. Wikipedia allows for 5000 requests per hour.
ONE_HOUR = 60 * 60

# Fetch the page contents
@sleep_and_retry
@limits(calls=5000, period=ONE_HOUR)
def get_page_content(page: str, verbose: bool = False) -> str:
  """Returns the content of a Wikipedia page."""
  url = f'https://api.wikimedia.org/core/v1/wikipedia/{LANGUAGE}/page/{page}/html'
  response = requests.get(url, headers=HEADERS)
  if response.status_code == 403:
    # If Wikipedia returns a 403 error, try it without the headers.
    if verbose: print('Returned 403.')
    response = requests.get(url)
    if response.status_code != 200:
      return None
  return response.text

# Get the HTMLs from the dist if available.

batch = {}

file_exists = exists(f'data/html.{LANGUAGE}.json')
if file_exists:
  print('Loading from file...')
  with open(f'data/html.{LANGUAGE}.json') as json_file:
    batch = json.load(json_file)
else:
  print('Creating a new file...')
  with open(f'data/html.{LANGUAGE}.json', 'w') as outfile:
    json.dump({}, outfile)

# Save all HTMLs to drive
pages_left = [page for page in pages if page not in list(batch.keys())]
print(f'There are {len(pages_left)} pages left...')

for i in tqdm(range(len(pages_left))):
  content = get_page_content(pages_left[i])
  if content != None:
    batch[pages_left[i]] = content
  if i % 100 == 0:
    print(f'Saving the batch to file...')
    with open(f'data/html.{LANGUAGE}.json', 'w') as outfile:
      json.dump(batch, outfile)

with open(f'data/html.{LANGUAGE}.json', 'w') as outfile:
  json.dump(batch, outfile)

# Helper methods
# Strip the keywords from parentheses
def strip(keyword: str):
  """Strips the keyword from parentheses."""
  if keyword == None:
    return ""
  return re.sub('\ \(.*\)', '', keyword)

def discriminate_keyword(keyword: str, text: str, entities: set, concepts: set) -> None:
  """Decides the validity of a keyword."""
  # Strip the parentheses
  keyword = strip(keyword)
  # Remove keywords with less than 3 length
  if len(keyword) < 3:
    return
  # Remove keywords that do not occur in text
  if keyword not in text and keyword.lower() not in text:
    return
  # Make keyword an entity if it has numerics
  has_numbers = re.search("[0-9]+", keyword)
  if has_numbers:
    entities.add(keyword)
    return
  # Make keyword an entity if it is not capitalized or uppercased
  is_capitalized_in_text = re.search(f"\w\ {keyword}", text)
  if is_capitalized_in_text:
    entities.add(keyword)
    return
  # Make keyword a concept if it is lowercased in text
  is_lowercased_in_text = re.search(f"\w\ {keyword.lower()}", text)
  if is_lowercased_in_text:
    concepts.add(keyword.lower())
    return
  return

def is_in_text(concept: str, text: str):
  """Checks if the concept is in the text."""
  match = re.compile(f'(\W|^){concept}').findall(text)
  return len(match) > 0

def process_page_content(page: str, verbose: bool = False) -> str:
  """Processes the content of a Wikipedia page."""
  html = batch[page]
  soup = BeautifulSoup(html, 'html.parser')
  # Get paragraphs
  paragraphs = "\n".join(list(map(lambda x: x.text, soup.find_all('p'))))
  paragraphs = re.sub(r'\[[0-9 a-z]+\]\ ?', '', paragraphs)
  # Get all keywords
  keywords = list(set(map(lambda x: x.get('title'), soup.find_all('a'))))
  # print("keywords: \n ", keywords)
  # Create concepts and entities
  concepts, entities = set(), set()
  for keyword in keywords:
    discriminate_keyword(keyword, paragraphs, entities, concepts)
  # Parse sentences
  sentences = nltk.tokenize.sent_tokenize(paragraphs, language="turkish")
  # Initialize a dataset
  dataset = []
  # Populate keywords for each sentences
  for sentence in sentences:
    # Remove trailing spaces
    sentence = sentence.strip()
    # Sentence concepts
    lowercase_concepts = set(concept for concept in concepts if re.search(f"\w\ {concept}", sentence))
    uppercase_concepts = set(concept.capitalize() for concept in concepts if re.search(f"${concept.capitalize()}", sentence))
    sent_concepts = sorted(list(lowercase_concepts.union(uppercase_concepts)), key=len)
    # Sentence entities
    sent_entities = sorted(list(set(entity for entity in entities if entity in sentence and entity not in uppercase_concepts)), key=len)
    data = {
      "sentence": sentence,
      "concepts": sent_concepts,
      "entities": sent_entities,
      "page_key": page,
    }
    dataset.append(data)
  if verbose:
    print("Dataset: \n ", dataset)
  json_dataset = json.dumps(dataset, ensure_ascii=False)
  return json_dataset

# Build a batch of sentences with their keywords

# Define the fetch content strategy


def process_pages(pages: list, checkpoint: int = 0) -> list:
  """Processes the content of a list of Wikipedia pages."""
  process_batch = []
  
  if checkpoint != 0:
    print(f'Loading checkpoint from {checkpoint}...')
    with open(f'data/checkpoints/sentences.{LANGUAGE}.{checkpoint}.json') as json_file:
      process_batch = json.load(json_file)
  
  for i in tqdm(range(len(pages) - checkpoint)):
    page = pages[i + checkpoint]
    
    try:
      results = json.loads(process_page_content(page))
      process_batch.extend(results)
    except:
      print('Error loading page content: ', page)
    
    if i % 100 == 0 and i > 0:
      print(f'Saving the checkpoint')
      with open(f'data/checkpoints/sentences.{LANGUAGE}.{i + checkpoint}.json', 'w') as outfile:
        json.dump(process_batch, outfile)
  return process_batch

# Create the complete batch
complete_batch = process_pages(list(batch.keys()))

# Analyze the batch
n_concept, n_entity = 0, 0
for i in tqdm(range(len(complete_batch))):
  n_concept += len(complete_batch[i]['concepts'])
  n_entity += len(complete_batch[i]['entities'])
print('Total count of concepts vs. entities:\n Concept: ', n_concept, '\nEntity: ', n_entity)

# Create Dataframe
# Create a dataframe from the batch.
df = pd.DataFrame.from_records(complete_batch)
# Remove duplicate sentences
def remove_duplicate_sentences(df: pd.DataFrame) -> pd.DataFrame:
  """Removes duplicate sentences."""
  print(f'Shape before removing duplicates: ', df.shape)
  df = df.drop_duplicates(subset=['sentence'])
  print(f'Shape after removing duplicates: ', df.shape)
  return df

df = remove_duplicate_sentences(df)

def split_datasets(df: pd.DataFrame) -> tuple:
  """Splits the dataframe into training and test datasets."""
  n_pages = len(pages)
  n_train, n_test = int(n_pages / 10 * 8), int(n_pages / 10)
  p_train, p_test, p_eval = pages[0:n_train], pages[n_train+1:-n_test], pages[(-n_test+1):-1]
  return p_train, p_test, p_eval

p_train, p_test, p_eval = split_datasets(df)

# Create seperate datasets and save them as files.
train_series = df['page_key'].isin(p_train)
train_df = df[train_series]
print(f'Shape of train set: ', train_df.shape)

test_series = df['page_key'].isin(p_test)
test_df = df[test_series]
print(f'Shape of test set: ', test_df.shape)

eval_series = df['page_key'].isin(p_eval)
eval_df = df[eval_series]
print(f'Shape of eval set: ', eval_df.shape)

train_df.to_json(f'wikikey-{LANGUAGE}-train.jsonl', orient='records', lines=True, force_ascii=False)
eval_df.to_json(f'wikikey-{LANGUAGE}-eval.jsonl', orient='records', lines=True, force_ascii=False)
test_df.to_json(f'wikikey-{LANGUAGE}-test.jsonl', orient='records', lines=True, force_ascii=False)

# Create Dataset
datasets = load_dataset('json', data_files={
  "train": f"wikikey-{LANGUAGE}-train.jsonl",
  "eval": f"wikikey-{LANGUAGE}-eval.jsonl",
  "test": f"wikikey-{LANGUAGE}-test.jsonl",
})
datasets.remove_columns(['page_key'])

if USE_HUGGINGFACE == True:
  datasets.push_to_hub(f"wikikey-{LANGUAGE}")
else:
  datasets.save_to_disk(f"datasets/wikikey-{LANGUAGE}")

# Postprocessing
if CONVERT_NUMBERIC_TO_ENTITY == True:
  hasNumber = re.compile('[0-9]+')

  def postprocess(sample):
    numeric_concepts = [concept for concept in sample['concepts'] if len(hasNumber.findall(concept)) > 0]
    if len(numeric_concepts) > 0:
      print(numeric_concepts)
      for numeric_concept in numeric_concepts:
        sample['concepts'].remove(numeric_concept)
        sample['entities'].append(numeric_concept)
    
    return sample

  datasets.map(postprocess)