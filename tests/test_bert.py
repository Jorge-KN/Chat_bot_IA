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
    # Contexto y pregunta sobre fotosíntesis
    context = (
        "La fotosíntesis es el proceso por el cual las plantas, las algas y algunas bacterias "
        "convierten la luz solar, el dióxido de carbono y el agua en glucosa y oxígeno. Este "
        "proceso ocurre en los cloroplastos, que contienen un pigmento verde llamado clorofila. "
        "La fotosíntesis es esencial para la vida en la Tierra, ya que proporciona el oxígeno "
        "necesario para la respiración y es la base de la cadena alimentaria."
    )
    question = "¿Dónde ocurre la fotosíntesis?"

    # Cargar el modelo entrenado
    tokenizer, model = load_model()

    # Obtener la respuesta
    answer = answer_question(context, question, tokenizer, model)

    # Mostrar los resultados
    print(f"Contexto: {context}")
    print(f"Pregunta: {question}")
    print(f"Respuesta: {answer}")
