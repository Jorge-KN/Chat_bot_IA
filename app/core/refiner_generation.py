import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Cargar el modelo y tokenizer entrenado
def load_correction_model(model_path="./models/modelo_corrector"):
    """Carga el modelo de corrección de texto"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, tokenizer, device

# Función para validar correcciones específicas
def validate_correction(original, corrected):
    """Valida si la corrección debe ser aceptada enfocándose en casos específicos"""
    if original == corrected:
        return original, False
    
    # Detectar si comienza con letra o número suelto seguido de espacio
    starts_with_single = re.match(r'^([a-zA-Z0-9])\s+(.+)$', original)
    if starts_with_single:
        expected_correction = starts_with_single.group(2)
        if corrected.strip() == expected_correction.strip():
            return corrected, True
    
    # Detectar números innecesarios en medio de texto
    has_unnecessary_number = re.search(r'\s+\d+\s+', original)
    if has_unnecessary_number:
        number_removed = re.sub(r'\s+\d+\s+', ' ', original)
        if corrected.strip() == number_removed.strip():
            return corrected, True
    
    return original, False

# Función para corregir texto usando el modelo cargado
def correct_text(text, model, tokenizer, device):
    """Corrige el texto usando el modelo y validación"""
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(
            **inputs,
            max_length=200,
            no_repeat_ngram_size=3,
            num_beams=5,
            early_stopping=True
        )
        corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return validate_correction(text, corrected)

# Función para integrar con el flujo principal
def process_and_correct(text, model_path="./models/modelo_corrector"):
    """Procesa un contexto y aplica correcciones según los modelos"""
    model, tokenizer, device = load_correction_model(model_path)
    corrected, was_corrected = correct_text(text, model, tokenizer, device)
    return corrected, was_corrected
