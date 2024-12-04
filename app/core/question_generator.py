import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizerFast, BertForQuestionAnswering
import random

# Cargar el modelo y el tokenizer entrenado para T5
t5_model_path = "./models/pregunta_model"  # Ruta al modelo T5 entrenado
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path)

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