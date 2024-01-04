import json

from requests import patch
from chatbot import get_response, predict_class
intents = json.loads(open('intents.json').read())

@patch('builtins.input', return_value="hola")  # Simular la entrada "hola"
def test_saludo(self, mock_input):
    sentence = "hola"
    tag = predict_class(sentence)
    assert tag == "saludo"

def test_saludo_respuesta():
    sentence = "hola"
    tag = predict_class(sentence)
    response = get_response(tag, intents)
    assert response == "Hola!"

@patch('builtins.input', return_value="dime tu nombre")
def test_nombre(self, mock_input):
    sentence = "dime tu nombre"
    tag = predict_class(sentence)
    assert tag == "nombre"

def test_nombre_respuesta(self, mock_input):
    sentence = "dime tu nombre"
    tag = predict_class(sentence)
    response = get_response(tag, intents)
    assert response in ["Soy el super bot de Cloud con Python", "Me llamo Pipi y soy un bot de IA con Python", "Me llamo Pipi de Python", "Soy un chatbot de Python sin nombre aun", "Soy Pipi, estoy entrenado para resolver tus consultas"]


print(test_saludo())
print(test_nombre_respuesta())
print(test_saludo_respuesta())