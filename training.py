import random
import json
import pickle
import numpy as np
import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.optimizers import *

lemmatizer = WordNetLemmatizer

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# loop through every intent
for intent in intents['intents']:
    # for each pattern in the intent
    for pattern in intent['patterns']:
        # split into individual words
        word_list = nltk.word_tokenize(pattern)

        # add to the words list
        words.append(word_list)

        # add tupple of the word list and the respective intent tag class
        documents.append((word_list, intent['tag']))

        # check if the class is in the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(documents)