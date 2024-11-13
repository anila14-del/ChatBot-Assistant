import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents
data_file = open('intents.json').read()
intents = json.loads(data_file)

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Tokenize each pattern and append to lists
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)  # Tokenize each word
        words.extend(w)  # Add words to words list
        documents.append((w, intent['tag']))  # Add pattern and associated tag

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))  # Remove duplicates and sort the words
classes = sorted(list(set(classes)))  # Sort the class names

print(f"{len(documents)} documents")
print(f"{len(classes)} classes: {classes}")
print(f"{len(words)} unique lemmatized words: {words}")

# Save the words and classes to pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

# Create the bag of words for each document
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # Create the bag of words (1 if word is in the pattern, else 0)
    for w in words:
        bag.append(1 if w in pattern_words else 0)
    
    # Create the output row (0 for each class, 1 for the current class)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    # Add to training set
    training.append([bag, output_row])

# Shuffle the training set and convert to numpy arrays
random.shuffle(training)

# Separate the training data into X (inputs) and Y (labels)
train_x = []
train_y = []

# Iterate through each document in the training set
for item in training:
    train_x.append(item[0])  # First element (bag of words)
    train_y.append(item[1])  # Second element (output row)

# Convert the lists to numpy arrays
train_x = np.array(train_x)
train_y = np.array(train_y)

print("Training data created")
print("Training data created")

# Create the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model using SGD with an updated syntax
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model.h5')

print("Model created")
