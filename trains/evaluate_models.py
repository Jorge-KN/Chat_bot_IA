import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizerFast, BertForQuestionAnswering
import random

#Este codigo no se ocupo para las pruebas finales

# Cargar el modelo y el tokenizer entrenado para T5
t5_model_path = "../models/pregunta_model"  # Ruta al modelo T5 entrenado
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path)

# Cargar el modelo y el tokenizer entrenado para BERT
bert_model_path = "../models/bert_model"  # Ruta al modelo BERT entrenado
bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)
bert_model = BertForQuestionAnswering.from_pretrained(bert_model_path)

# Función para generar una pregunta con T5
def generate_question(context_text):
    input_text = f"Contexto: {context_text}. Genera una pregunta sobre este contexto."
    input_ids = t5_tokenizer(input_text, return_tensors="pt").input_ids

    # Generar la pregunta
    outputs = t5_model.generate(
        input_ids,
        max_length=80,
        num_beams=20,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        early_stopping=True
    )

    # Decodificar la pregunta generada
    question = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question if question else "No se pudo generar una pregunta."

# Función para responder a una pregunta con BERT
def answer_question(context, question, tokenizer, model):
    # Tokenizar el contexto y la pregunta
    inputs = tokenizer(
        question,
        context,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    # Realizar la inferencia
    with torch.no_grad():
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Obtener las posiciones de inicio y fin con mayor probabilidad
        start_idx = torch.argmax(start_logits)
        end_idx = torch.argmax(end_logits) + 1

    # Extraer la respuesta
    input_ids = inputs["input_ids"].squeeze()
    answer_tokens = input_ids[start_idx:end_idx]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer

# Función para generar respuestas incorrectas aleatorias
def generate_incorrect_answers(context, correct_answer):
    sentences = context.split('.')
    # Eliminar la respuesta correcta de las opciones posibles
    incorrect_answers = []
    while len(incorrect_answers) < 3:
        # Selecciona una frase aleatoria que no sea la respuesta correcta
        random_sentence = random.choice([s.strip() for s in sentences if s.strip() not in incorrect_answers and s.strip() != correct_answer])
        if random_sentence and random_sentence not in incorrect_answers:
            incorrect_answers.append(random_sentence)
    return incorrect_answers

# Contexto de prueba
test_context = (
    "La fotosíntesis es el proceso mediante el cual las plantas verdes producen su alimento. "
    "Utilizando la luz solar, el dióxido de carbono y el agua, las plantas generan glucosa y liberan oxígeno. "
    "Este proceso es fundamental para la vida en la Tierra, ya que proporciona oxígeno a la atmósfera."
)

# Paso 1: Generar pregunta usando el modelo T5
generated_question = generate_question(test_context)
print(f"Pregunta generada: {generated_question}")

# Paso 2: Responder a la pregunta utilizando el modelo BERT
correct_answer = answer_question(test_context, generated_question, bert_tokenizer, bert_model)
print(f"Respuesta correcta generada por BERT: {correct_answer}")

# Paso 3: Generar respuestas incorrectas
incorrect_answers = generate_incorrect_answers(test_context, correct_answer)

# Mezclar la respuesta correcta con las incorrectas
options = [correct_answer] + incorrect_answers
random.shuffle(options)


# Mostrar las opciones de respuesta de selección múltiple
print("\nOpciones de respuesta:")
for i, option in enumerate(options, 1):
    print(f"{i}. {option}")