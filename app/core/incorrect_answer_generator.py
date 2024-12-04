#
#import random
#
## Función mejorada para generar respuestas incorrectas basadas en el contexto y la longitud de la respuesta correcta
#def generate_incorrect_answers(context, correct_answer):
#    sentences = context.split('. ')  # Dividir el contexto en oraciones
#    incorrect_answers = []
#
#    # Longitud objetivo basada en la respuesta correcta
#    target_length = len(correct_answer)
#
#    # Filtrar frases irrelevantes y eliminar la respuesta correcta
#    candidates = [
#        s.strip() for s in sentences 
#        if correct_answer not in s.strip() and abs(len(s.strip()) - target_length) <= 10  # Filtrar por longitud cercana
#    ]
#
#    # Si hay suficientes candidatos, selecciona aleatoriamente
#    if len(candidates) >= 3:
#        incorrect_answers = random.sample(candidates, 3)
#    else:
#        # Usar todas las frases disponibles
#        incorrect_answers = candidates
#
#    # Si no hay suficientes, generar distracciones genéricas desde el contexto
#    while len(incorrect_answers) < 3:
#        distractor = random.choice(sentences).strip()
#        if distractor not in incorrect_answers and correct_answer not in distractor:
#            incorrect_answers.append(distractor[:target_length])
#
#    return incorrect_answers[:3]  # Asegurarse de devolver exactamente 3 respuestas

import random
import nltk
import stanza
import re
import logging
import warnings

# Ignorar las advertencias de tipo FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Configurar el logger de 'stanza' para que solo muestre errores críticos
logging.getLogger('stanza').setLevel(logging.ERROR)
stanza.download('es', verbose=False)

# Inicializar el pipeline sin mensajes adicionales
nlp = stanza.Pipeline('es', verbose=False)
# Descargar recursos necesarios de NLTK solo si no están disponibles
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Inicializar Stanza y descargar el modelo en español
stanza.download('es')  # Solo es necesario ejecutarlo una vez
nlp = stanza.Pipeline('es', processors='tokenize,pos', tokenize_pretokenized=False)

# Función para extraer sustantivos del contexto usando Stanza
def extract_nouns(context):
    doc = nlp(context)
    nouns = []
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.upos == 'NOUN':
                nouns.append(word.text.lower())
    nouns = list(set(nouns))
    return nouns

# Función principal para generar respuestas incorrectas
def generate_incorrect_answers(context, correct_answer):
    # Extraer sustantivos del contexto
    nouns = extract_nouns(context)
    # Remover palabras de la respuesta correcta de los sustantivos
    correct_answer_words = correct_answer.lower().split()
    nouns = [noun for noun in nouns if noun not in correct_answer_words]
    # Si hay menos de 3 sustantivos, agregar palabras genéricas
    if len(nouns) < 3:
        generic_words = ["concepto", "idea", "definición", "proceso", "teoría"]
        nouns.extend(generic_words)
        nouns = list(set(nouns))
    # Seleccionar aleatoriamente 3 sustantivos como respuestas incorrectas
    if len(nouns) >= 3:
        incorrect_answers = random.sample(nouns, 3)
    else:
        incorrect_answers = nouns
    return incorrect_answers