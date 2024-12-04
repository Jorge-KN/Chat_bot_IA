import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

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

# Lista de contextos para las pruebas
test_contexts = [
    "El Sol es una estrella situada en el centro del sistema solar. Proporciona la luz y el calor necesarios para mantener la vida en la Tierra.",
    "El agua se compone de dos átomos de hidrógeno y uno de oxígeno, formando la fórmula H₂O. Es esencial para la vida y cubre el 71% de la superficie terrestre.",
    "Isaac Newton fue un físico británico que formuló las leyes del movimiento y la ley de la gravitación universal.",
    "La Gran Muralla China es una antigua fortificación construida para proteger el territorio chino de invasiones.",
    "El ciclo del agua incluye la evaporación, condensación y precipitación, asegurando la distribución del agua dulce en la Tierra.",
    "El Amazonas es el río más largo del mundo y tiene la mayor cuenca hidrográfica, atravesando Brasil, Perú y Colombia.",
    "Los dinosaurios se extinguieron hace aproximadamente 66 millones de años debido a un posible impacto de asteroide.",
    "Albert Einstein desarrolló la teoría de la relatividad y recibió el Premio Nobel de Física en 1921 por su explicación del efecto fotoeléctrico.",
    "La contaminación del aire es causada por sustancias como el dióxido de carbono y los óxidos de nitrógeno, afectando la salud humana.",
    "El Everest es la montaña más alta del mundo, situada en la cordillera del Himalaya, con una altura de 8,848 metros."
]

# Generar preguntas para cada contexto
for i, context in enumerate(test_contexts, 1):
    question = generate_question(context)
    print(f"Contexto {i}: {context}")
    print(f"Pregunta generada: {question}")
    print("-" * 50)
