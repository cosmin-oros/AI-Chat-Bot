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
from tensorflow.python.keras.optimizers import gradient_descent_v2

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

# sequential model for neural network
model = Sequential()
# input layer with 128 neurons, activation rectified linear unit
model.add(Dense(128, input_shape=(len(training_x[0]),), activation='relu'))
# prevent over fitting
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# softmax scales the results so that all add up to 1
model.add(Dense(len(training_y[0]), activation='softmax'))

sgd = gradient_descent_v2.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# compile the model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(training_x), np.array(training_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.model')

print("Done")