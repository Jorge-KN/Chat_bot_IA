from transformers import T5ForConditionalGeneration, T5Tokenizer

# Cargar el modelo T5
def load_data(data):
    # Aquí podrías agregar la lógica para procesar el archivo de datos
    # Supongamos que los datos son un diccionario de tipo JSON
    return {'status': 'Data loaded successfully'}

def generate_question(content):
    # Cargar el tokenizer y el modelo T5
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    # Preprocesar el texto
    input_text = f"generate question: {content}"
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generar la pregunta
    outputs = model.generate(inputs['input_ids'])
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return question
