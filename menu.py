from app.core.answer_generator import *
from app.core.question_generator import *
from app.core.incorrect_answer_generator import *
from app.core.refiner_generation import *
import json
import random
import re

# Variable global para guardar el libro seleccionado
selected_book = None

# Función para eliminar guiones que separan palabras
def remove_word_hyphenation(text):
    # Eliminar guiones entre palabras y unirlas sin espacio
    return re.sub(r"(?<=\w)\s*-\s*(?=\w)", "", text)

# Función para cargar contextos desde un archivo JSON con validaciones
def load_contexts_from_json(json_path="./json_outputs/processed_data.json", min_length=70, valid_starts=None):
    global selected_book  # Utilizar la variable global para guardar el libro seleccionado

    if valid_starts is None:
        valid_starts = ["el", "la", "un", "una", "los", "las"]  # Palabras de inicio predeterminadas

    # Convertir las palabras válidas a minúsculas para comparación insensible a mayúsculas/minúsculas
    valid_starts = [word.lower() for word in valid_starts]

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Mostrar los libros disponibles solo si no hay un libro previamente seleccionado
    if selected_book is None:
        print("\nLibros disponibles:")
        for idx, book in enumerate(data.keys(), 1):
            print(f"{idx}. {book}")

        # Selección del libro por número
        try:
            choice = int(input("\nElige el libro del cual deseas extraer el texto (número): "))
            selected_book = list(data.keys())[choice - 1]
        except (ValueError, IndexError):
            raise ValueError("Selección inválida, por favor elige un número de la lista.")

    # Procesar los párrafos del libro seleccionado
    all_paragraphs = []
    excluded_phrases = ["http://", "https://", "Recuperado de", "www."]

    for paragraph in data[selected_book]:
        # Aplicar el filtro para eliminar guiones
        paragraph = remove_word_hyphenation(paragraph)

        # Verificar longitud mínima
        if len(paragraph) < min_length:
            continue
        # Excluir contextos que contengan enlaces o referencias
        if any(exclusion in paragraph for exclusion in excluded_phrases):
            continue
        # Verificar que el contexto comience con una palabra válida
        first_word = paragraph.split()[0].lower()
        if first_word not in valid_starts:
            continue
        all_paragraphs.append(paragraph)

    if not all_paragraphs:
        raise ValueError(f"No se encontraron contextos válidos con al menos {min_length} caracteres en el libro '{selected_book}'.")

    return random.choice(all_paragraphs)

# Función principal para generar pregunta y opciones
def generate_question_and_answers():
    # Cargar contexto desde el archivo JSON
    json_path = "./json_outputs/processed_data.json"  # Cambia esta ruta si es necesario
    test_context = load_contexts_from_json(json_path)
    print(f"\nContexto seleccionado:\n{test_context}")

    # Corregir el contexto utilizando process_and_correct
    corrected_context, was_corrected = process_and_correct(test_context)
    if was_corrected:
        print(f"\nContexto corregido:\n{corrected_context}")
    else:
        print("\nNo se realizaron correcciones al contexto.")

    # Utilizar el contexto corregido o el original
    context_to_use = corrected_context if was_corrected else test_context

    # Paso 1: Generar pregunta usando el modelo T5
    generated_question = generate_question(context_to_use)
    print(f"\nPregunta generada: {generated_question}")

    # Paso 2: Responder a la pregunta utilizando el modelo BERT
    correct_answer = answer_question(context_to_use, generated_question, bert_tokenizer, bert_model)
    print(f"\nRespuesta correcta generada: {correct_answer}")

    # Paso 3: Generar respuestas incorrectas
    incorrect_answers = generate_incorrect_answers(context_to_use, correct_answer)

    # Mezclar la respuesta correcta con las incorrectas
    options = [correct_answer] + incorrect_answers
    random.shuffle(options)

    # Mostrar las opciones de respuesta de selección múltiple
    print("\nOpciones de respuesta:")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")

# Menú principal
def main_menu():
    while True:
        print("\n--- Menú Principal ---")
        print("1. Generar pregunta")
        print("2. Salir")
        choice = input("\nElige una opción: ")

        if choice == "1":
            sub_menu()
        elif choice == "2":
            print("\nSaliendo del programa...")
            break
        else:
            print("\nOpción no válida, intenta de nuevo.")

# Submenú para continuar o salir
def sub_menu():
    while True:
        generate_question_and_answers()
        print("\n--- Submenú ---")
        print("1. Generar otra pregunta")
        print("2. Regresar al menú principal")
        print("3. Salir")
        choice = input("\nElige una opción: ")

        if choice == "1":
            continue
        elif choice == "2":
            global selected_book
            selected_book = None
            break
        elif choice == "3":
            print("\nSaliendo del programa...")
            exit()
        else:
            print("\nOpción no válida, intenta de nuevo.")

if __name__ == "__main__":
    main_menu()
