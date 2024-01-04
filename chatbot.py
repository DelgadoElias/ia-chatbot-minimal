import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    # De todas las palabras de aca, si una de las que introduce el usuario coincide, agrega un uno a la bolsa de palabras
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    # Devuelve esta bolsa de palabras
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    # Ese [0] es por como funciona el predict como array
    res = model.predict(np.array([bow]))[0]
    # Este resultado devuelve una probabilidad de a que lugar pertenece la frase seleccionada

    # Lo que necesitamos es usar el de la probabilidad mas alta, por eso esta linea
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category

def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i["tag"]==tag:
            result = random.choice(i['responses'])
            break
    return result

while True:
    message= input("")

    # Salir del bucle si el usuario ingresa 'exit' o presiona Enter sin ingresar nada
    if message.lower() == 'exit' or message == '':
        print("Goodbye!")
        break

    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)