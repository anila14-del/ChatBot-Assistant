import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np

lemmatizer = WordNetLemmatizer()
ignore_words = ['?', '!']

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, ignore_words=ignore_words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model, words, classes):
    p = bow(sentence, words)
    p = np.array([p])
    prediction = model.predict(p)
    return classes[np.argmax(prediction)]

def response_to_intent(intent):
    responses = {
        'greeting': 'Hello! How can I assist you today?',
        'goodbye': 'Goodbye! Have a nice day!',
        'thanks': 'You are welcome!',
        # Add other responses as necessary
    }
    return responses.get(intent, "Sorry, I didn't understand that.")