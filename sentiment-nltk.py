import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



plt.style.use('ggplot')

import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Read in data
data = pd.read_csv('data.csv')

# print(data.head())
# print(data['description'].values[0])


#Example

# example = data['description'].values[3]
# print(example)

# d_tokens = nltk.word_tokenize(example)
# print(d_tokens)

# tagged = nltk.pos_tag(d_tokens)
# print(tagged)

# chunks = nltk.chunk.ne_chunk(tagged)
# chunks.pprint()

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# text1 = 'I am so happy with Sabre support'
# text2 = 'This is the worst support'
# sentiment1 = sia.polarity_scores(text1)
# sentiment2 = sia.polarity_scores(text2)
# sentiment3 = sia.polarity_scores(example)
# print(sentiment1)
# print(sentiment2)
# print(sentiment3)

# Results
results = {}
# Polarity score on data
for  i, row in tqdm(data.iterrows(), total=len(data)):
    text = row['description']
    caseID = row['case number']
    results[caseID] = sia.polarity_scores(text)

print(results)


