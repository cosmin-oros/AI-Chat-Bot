import random
import json
import pickle
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.optimizers import *

lemmatizer = WordNetLemmatizer()

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
        words.extend(word_list)

        # add tupple of the word list and the respective intent tag class
        documents.append((word_list, intent['tag']))

        # check if the class is in the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
# eliminate dups
words = sorted(set(words))

classes = sorted(set(classes))

# save them into a file for later on
pickle.dump(words, open('words.pk1', 'wb'))
pickle.dump(words, open('classes.pk1', 'wb'))

training = []
output_empty = [0]*len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# shuffle the data
random.shuffle(training)
training = np.array(training)

training_x = list(training[:, 0])
training_y = list(training[:, 1])