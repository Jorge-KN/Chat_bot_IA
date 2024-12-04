import json
import random
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
import difflib
import re

nltk.download('punkt')
nltk.download('punkt_tab')


def load_model(model_path="../models/modelo_corrector"):
    """Carga el modelo y tokenizer guardados"""
    print("Cargando modelo y tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, tokenizer, device

def validate_correction(original, corrected):
    """Valida si la corrección debe ser aceptada enfocándose en casos específicos"""
    if original == corrected:
        return original, False
    
    # Detectar si comienza con letra o número suelto seguido de espacio
    starts_with_single = re.match(r'^([a-zA-Z0-9])\s+(.+)$', original)
    if starts_with_single:
        # Si la corrección elimina ese carácter inicial y mantiene el resto
        expected_correction = starts_with_single.group(2)
        if corrected.strip() == expected_correction.strip():
            return corrected, True
    
    # Detectar números innecesarios en medio de texto
    has_unnecessary_number = re.search(r'\s+\d+\s+', original)
    if has_unnecessary_number:
        # Verificar que la corrección elimina el número manteniendo el contexto
        number_removed = re.sub(r'\s+\d+\s+', ' ', original)
        if corrected.strip() == number_removed.strip():
            return corrected, True
    
    # Mantener el original en otros casos
    return original, False

def correct_text(text, model, tokenizer, device):
    """Corrige el texto usando el modelo y validación"""
    # Primero intentar corrección manual para casos específicos
    if re.match(r'^[kK]\s+', text):  # Si comienza con 'k' o 'K' seguido de espacio
        manual_correction = text[2:].strip()
        return manual_correction, True
    
    if re.match(r'^[a-zA-Z0-9]\s+', text):  # Si comienza con letra/número suelto
        manual_correction = text[2:].strip()
        return manual_correction, True
    
    # Si no aplica corrección manual, usar el modelo
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

def filter_sentences(sentences):
    """Filtra oraciones que necesitan corrección según criterios específicos"""
    filtered = []
    for sentence in sentences:
        # Comienza con letra/número suelto
        if re.match(r'^[a-zA-Z0-9]\s+', sentence):
            filtered.append(sentence)
            continue
            
        # Tiene números innecesarios en medio del texto
        if re.search(r'\s+\d+\s+', sentence):
            filtered.append(sentence)
            continue
            
    return filtered

def get_contexts():
    """Retorna una lista de contextos predefinidos"""
    return [
        "k Esto es un texto que comienza con un error típico.",
        "A simple 123 texto con números aleatorios en medio.",
        "1 El número uno se presenta innecesariamente al inicio.",
        "Un párrafo 456 con un número sin propósito en el medio.",
        "z Esto es una frase que inicia con una letra suelta.",
        "7 Esto es un ejemplo donde un número inicial no tiene relevancia.",
        "Texto válido que sin embargo incluye 789 números innecesarios.",
        "C Esto es un inicio con una sola letra, que no tiene sentido.",
        "Una oración donde el número 33 aparece sin razón.",
        "H Este texto empieza con una letra que debería eliminarse."
    ]

def main():
    try:
        model, tokenizer, device = load_model()
        print(f"Usando dispositivo: {device}")
        
        # Cargar contextos predefinidos
        sentences = get_contexts()
        
        print("\nProcesando oraciones seleccionadas:\n")
        for i, sentence in enumerate(sentences, 1):
            print(f"\nOración {i}:")
            print(f"Original ({len(sentence)} caracteres):\n{sentence}")
            
            corrected, was_corrected = correct_text(sentence, model, tokenizer, device)
            print(f"Corregida ({len(corrected)} caracteres):\n{corrected}")
            
            if was_corrected:
                differences = list(difflib.ndiff(sentence, corrected))
                changes = [d for d in differences if d[0] != ' ']
                if changes:
                    print("\nCambios realizados:")
                    for change in changes:
                        if change[0] == '-':
                            print(f"  - Eliminado: '{change[2:]}'")
                        elif change[0] == '+':
                            print(f"  + Añadido: '{change[2:]}'")
                print("(Se realizaron correcciones validadas)")
            else:
                print("(No requiere correcciones)")
            print("-" * 80)

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
