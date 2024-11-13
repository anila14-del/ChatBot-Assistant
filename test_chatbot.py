import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
from keras.models import load_model
import pickle
from chatbot_utils import clean_up_sentence, bow, predict_class, response_to_intent  # Absolute import

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load model and other necessary files
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Define a function to get the bot's response
def chatbot_response(text):
    # Predict class based on input
    intent = predict_class(text, model, words, classes)

    # Get the bot's response to the predicted intent
    return response_to_intent(intent)

# Start a loop to get user input
print("Chatbot is running! Type 'exit' to stop the chat.")
while True:
    # Get user input
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        break
    
    # Get and display the chatbot's response
    response = chatbot_response(user_input)
    print(f"Bot: {response}")