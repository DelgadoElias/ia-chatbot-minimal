import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

# Clasifica los patrones y las categorías
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Pasa la información a unos y ceros según las palabras presentes en cada categoría para hacer el entrenamiento
training = []
output_empty = [0] * len(classes)
# Inicializa las listas para las bolsas (bags) y las salidas (outputs)
bags = []
outputs = []

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    # Crea una bolsa (bag) con la misma longitud que la lista de palabras (words)
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # Asegúrate de que la longitud de la bolsa sea igual a la longitud de classes
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    bags.append(bag)
    outputs.append(output_row)

# Convierte las listas en arrays de NumPy
bags = np.array(bags)
outputs = np.array(outputs)

# Combina las bolsas y las salidas en un solo array
training = np.column_stack((bags, outputs))

random.shuffle(training)
training = np.array(training) 
print(training)

# Ahora que tenemos esta "matriz de entrenamiento" lo dividimos en dos variables para más comprensión
train_x = np.array(training[:, :len(words)], dtype=np.float32)
train_y = np.array(training[:, len(words):], dtype=np.float32)



# Crearemos una red neuronal secuencial
model = Sequential()
model.add(Dense(128, input_dim=len(train_x[0]), activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(len(classes), activation='softmax'))

# Propiedades a colocar para que el optimizador funcione correctamente en la mayoría de los casos
sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

train_process = model.fit(train_x, train_y, epochs=1000000, batch_size=6, verbose=1)

model.save("chatbot_model.h5", train_process)


