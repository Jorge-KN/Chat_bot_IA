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

def process_json_file(json_path="../json_outputs/processed_data.json", min_chars=50, max_chars=100, num_sentences=5):
    """Procesa el archivo JSON y extrae oraciones que necesitan corrección"""
    print(f"Cargando archivo JSON: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    all_paragraphs = []
    for key, paragraphs in data.items():
        all_paragraphs.extend(paragraphs)
    
    print("Procesando oraciones...")
    candidates = []
    
    for paragraph in tqdm(all_paragraphs, desc="Buscando oraciones para corregir"):
        sentences = nltk.sent_tokenize(paragraph)
        # Filtrar por longitud y luego por criterios específicos
        length_filtered = [s.strip() for s in sentences if min_chars <= len(s.strip()) <= max_chars]
        candidates.extend(filter_sentences(length_filtered))
    
    print(f"\nSe encontraron {len(candidates)} oraciones que necesitan corrección")
    
    if not candidates:
        raise ValueError("No se encontraron oraciones que cumplan los criterios de corrección")
    
    selected = random.sample(candidates, min(num_sentences, len(candidates)))
    return selected

def main():
    try:
        model, tokenizer, device = load_model()
        print(f"Usando dispositivo: {device}")
        
        sentences = process_json_file()
        
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