import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering

# Cargar el modelo y el tokenizador entrenados
def load_model(model_path="../models/bert_model"):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForQuestionAnswering.from_pretrained(model_path)
    return tokenizer, model

# Responder una pregunta basada en un contexto
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

if __name__ == "__main__":
    # Lista de contextos y preguntas
    test_cases = [
        {
            "context": "El Sol es una estrella situada en el centro del sistema solar. Proporciona la luz y el calor necesarios para mantener la vida en la Tierra. La energía solar es aprovechada por las plantas a través de la fotosíntesis y también puede ser utilizada por los seres humanos mediante paneles solares.",
            "question": "¿Qué es el Sol?"
        },
        {
            "context": "El agua es una sustancia esencial para todos los seres vivos. Se compone de dos átomos de hidrógeno y uno de oxígeno, formando la fórmula H₂O. En su estado líquido, el agua es transparente e inodora. Es el principal componente de los océanos, ríos y lagos, y cubre aproximadamente el 71% de la superficie terrestre.",
            "question": "¿Cuál es la fórmula química del agua?"
        },
        {
            "context": "Isaac Newton fue un físico, matemático y astrónomo británico, considerado uno de los científicos más influyentes de todos los tiempos. Su obra 'Philosophiæ Naturalis Principia Mathematica' sentó las bases de la mecánica clásica. Entre sus descubrimientos más notables se encuentran las leyes del movimiento y la ley de la gravitación universal.",
            "question": "¿Qué libro escribió Isaac Newton?"
        },
        {
            "context": "La Gran Muralla China es una antigua fortificación construida para proteger el territorio chino de invasiones y ataques. Su construcción comenzó en el siglo III a. C. y continuó durante varias dinastías. Se extiende por más de 21,000 kilómetros y es considerada una de las maravillas del mundo moderno.",
            "question": "¿Cuál era el propósito de la Gran Muralla China?"
        },
        {
            "context": "El ciclo del agua es el proceso mediante el cual el agua se mueve constantemente entre la superficie de la Tierra y la atmósfera. Este ciclo incluye la evaporación, la condensación y la precipitación. Es un proceso clave para mantener la vida en el planeta, ya que asegura la distribución del agua dulce.",
            "question": "¿Cuáles son las etapas del ciclo del agua?"
        },
        {
            "context": "El Amazonas es el río más largo del mundo y tiene la mayor cuenca hidrográfica. Atraviesa varios países de América del Sur, incluyendo Brasil, Perú y Colombia. Su biodiversidad es extraordinaria, y alberga una gran variedad de especies animales y vegetales.",
            "question": "¿Qué países atraviesa el Amazonas?"
        },
        {
            "context": "Los dinosaurios fueron un grupo de reptiles que habitaron la Tierra durante el período Mesozoico, hace millones de años. Se extinguieron hace aproximadamente 66 millones de años debido a un evento catastrófico, posiblemente el impacto de un asteroide. Los dinosaurios incluían especies carnívoras y herbívoras.",
            "question": "¿Cuándo se extinguieron los dinosaurios?"
        },
        {
            "context": "Albert Einstein fue un físico teórico nacido en Alemania, conocido por desarrollar la teoría de la relatividad. Su fórmula E=mc², que establece la equivalencia entre masa y energía, es una de las más famosas en la historia de la ciencia. Recibió el Premio Nobel de Física en 1921 por su explicación del efecto fotoeléctrico.",
            "question": "¿Por qué recibió Einstein el Premio Nobel de Física?"
        },
        {
            "context": "La contaminación del aire es un problema ambiental causado por la liberación de sustancias nocivas como el dióxido de carbono, el monóxido de carbono y los óxidos de nitrógeno. Estas sustancias provienen principalmente de actividades humanas como la quema de combustibles fósiles y la industria. La contaminación del aire puede causar problemas de salud, como enfermedades respiratorias.",
            "question": "¿Qué causa la contaminación del aire?"
        },
        {
            "context": "El Everest es la montaña más alta del mundo, con una altura de 8,848 metros sobre el nivel del mar. Se encuentra en la cordillera del Himalaya, en la frontera entre Nepal y China. Cada año, muchos alpinistas intentan alcanzar su cima, enfrentando condiciones climáticas extremas.",
            "question": "¿Dónde se encuentra el Everest?"
        }
    ]

    # Cargar el modelo entrenado
    tokenizer, model = load_model()

    # Procesar cada caso de prueba
    for i, case in enumerate(test_cases, start=1):
        context = case["context"]
        question = case["question"]
        answer = answer_question(context, question, tokenizer, model)
        print(f"Prueba {i}:")
        print(f"Contexto: {context}")
        print(f"Pregunta: {question}")
        print(f"Respuesta: {answer}")
        print("-" * 50)
