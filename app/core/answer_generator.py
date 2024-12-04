import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizerFast, BertForQuestionAnswering


#Cargar el modelo y el tokenizer entrenado para BERT
bert_model_path = "./models/bert_model"  # Ruta al modelo BERT entrenado
bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)
bert_model = BertForQuestionAnswering.from_pretrained(bert_model_path)

#Función para responder a una pregunta con BERT
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
