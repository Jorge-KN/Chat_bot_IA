import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import random

# Cargar el modelo y el tokenizer entrenado
model_path = "../models/pregunta_model"  # Ruta al modelo entrenado
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Función para generar una pregunta en base a un contexto dado
def generate_question(context_text):
    input_text = f"Contexto: {context_text}. Genera una pregunta sobre este contexto."
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # Generar la pregunta
    outputs = model.generate(
        input_ids,
        max_length=80,
        num_beams=20,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        early_stopping=True
    )

    # Decodificar la pregunta generada
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question if question else "No se pudo generar una pregunta."

# Función para generar las opciones de respuesta random como metodo de prueba
def generate_options(context_text, question):
    question_keywords = [word.lower() for word in question.split() if len(word) > 3 and word.isalpha()]
    relevant_words = [word for word in context_text.split() if word.lower() in question_keywords and len(word) > 3 and word.isalpha()]

    if not relevant_words:
        relevant_words = [word for word in context_text.split() if word.isalpha() and len(word) > 3]

    correct_answer = random.choice(relevant_words)

    distractors = [w for w in context_text.split() if w != correct_answer and len(w) > 3 and w.isalpha()]
    
    num_distractors = min(3, len(distractors))
    distractors = random.sample(distractors, num_distractors)

    options = distractors + [correct_answer]
    random.shuffle(options)

    return options, correct_answer

# Contexto de prueba
test_context = "La fotosíntesis es el proceso mediante el cual las plantas verdes producen su alimento. Utilizando la luz solar, el dióxido de carbono y el agua, las plantas generan glucosa y liberan oxígeno. Este proceso es fundamental para la vida en la Tierra, ya que proporciona oxígeno a la atmósfera."

# Generar pregunta y opciones
question = generate_question(test_context)
options, correct_answer = generate_options(test_context, question)

# Mostrar el resultado
print(f"Contexto: {test_context}")
print(f"Pregunta generada: {question}")
print("Opciones:")
for i, option in enumerate(options, 1):
    print(f"{i}. {option}")
print(f"Respuesta correcta: {correct_answer}")
