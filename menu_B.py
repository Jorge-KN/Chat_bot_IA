import tkinter as tk
from tkinter import filedialog, messagebox
import json
import random
import re
import os
from app.core.answer_generator import *
from app.core.question_generator import *
from app.core.incorrect_answer_generator import *
from app.core.refiner_generation import *
import fitz  # PyMuPDF

# Variables globales
books_data = None
selected_book = None
UPLOAD_FOLDER = './uploads'
JSON_FOLDER = './json_outputs'

# Crear carpetas necesarias si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(JSON_FOLDER, exist_ok=True)

# Función para eliminar guiones que separan palabras
def remove_word_hyphenation(text):
    return re.sub(r"(?<=\w)\s*-\s*(?=\w)", "", text)

# Cargar el archivo JSON al iniciar
def load_books_from_json(json_path="./json_outputs/processed_data.json"):
    """Carga los libros desde el archivo JSON."""
    global books_data
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            books_data = json.load(file)
        return list(books_data.keys())
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo cargar el archivo JSON: {e}")
        return []

# Actualizar el listado de libros en el Listbox
def update_books_listbox():
    """Actualiza el Listbox con los libros disponibles en el JSON."""
    books_listbox.delete(0, tk.END)  # Limpia el Listbox
    books = load_books_from_json()  # Carga los libros actualizados
    for book in books:
        books_listbox.insert(tk.END, book)

# Seleccionar libro
def select_book(event):
    """Selecciona el libro desde el Listbox."""
    global selected_book
    selection = books_listbox.curselection()
    if selection:
        selected_book = books_listbox.get(selection[0])

# Cargar contextos desde el libro seleccionado
def load_contexts_from_json(min_length=70, valid_starts=None):
    """Carga un contexto válido del libro seleccionado."""
    global selected_book, books_data

    if not selected_book:
        raise ValueError("No se ha seleccionado ningún libro.")

    if valid_starts is None:
        valid_starts = ["el", "la", "un", "una", "los", "las"]

    valid_starts = [word.lower() for word in valid_starts]

    all_paragraphs = []
    excluded_phrases = ["http://", "https://", "Recuperado de", "www."]

    for paragraph in books_data[selected_book]:
        paragraph = remove_word_hyphenation(paragraph)
        if len(paragraph) < min_length:
            continue
        if any(exclusion in paragraph for exclusion in excluded_phrases):
            continue
        first_word = paragraph.split()[0].lower()
        if first_word not in valid_starts:
            continue
        all_paragraphs.append(paragraph)

    if not all_paragraphs:
        raise ValueError(f"No se encontraron contextos válidos en el libro '{selected_book}'.")

    return random.choice(all_paragraphs)

# Generar pregunta y opciones
def generate_question_and_answers():
    """Genera una pregunta y opciones de respuesta."""
    try:
        test_context = load_contexts_from_json()
        corrected_context, was_corrected = process_and_correct(test_context)

        context_to_use = corrected_context if was_corrected else test_context

        generated_question = generate_question(context_to_use)
        correct_answer = answer_question(context_to_use, generated_question, bert_tokenizer, bert_model)
        incorrect_answers = generate_incorrect_answers(context_to_use, correct_answer)

        options = [correct_answer] + incorrect_answers
        random.shuffle(options)

        return context_to_use, generated_question, options
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None, None, None

# Mostrar pregunta y opciones
def display_question():
    """Muestra el contexto, la pregunta generada y las opciones en la interfaz."""
    if not selected_book:
        messagebox.showerror("Error", "Por favor, selecciona un libro de la lista.")
        return

    context, question, options = generate_question_and_answers()
    if context and question and options:
        context_label.config(text=f"Contexto:\n{context}")
        question_label.config(text=f"Pregunta:\n{question}")
        options_label.config(text="\nOpciones:\n" + "\n".join([f"{i+1}. {option}" for i, option in enumerate(options)]))

# Subir y procesar PDFs
def upload_and_process_pdfs():
    """Permite subir y procesar archivos PDF."""
    files = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])
    if not files:
        messagebox.showinfo("Info", "No se seleccionaron archivos.")
        return

    json_data = {}
    already_uploaded = []
    processed_files = []
    json_output_path = os.path.join(JSON_FOLDER, 'processed_data.json')

    # Cargar datos JSON existentes si ya hay un archivo
    if os.path.exists(json_output_path):
        with open(json_output_path, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)

    for file in files:
        filename = os.path.basename(file)
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        if os.path.exists(file_path):
            already_uploaded.append(filename)
            continue

        # Guardar el archivo en UPLOAD_FOLDER
        os.rename(file, file_path)

        # Extraer texto y preprocesar
        text = extract_text_from_pdf(file_path)
        preprocessed_text = preprocess_text(text)

        json_data[filename] = preprocessed_text
        processed_files.append(filename)

    # Guardar datos procesados en JSON
    with open(json_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)

    # Actualizar Listbox después de procesar los archivos
    update_books_listbox()

    messagebox.showinfo("Resultado", f"Archivos procesados: {processed_files}\nArchivos ya existentes: {already_uploaded}")

# Funciones auxiliares
def extract_text_from_pdf(file_path):
    """Extrae el texto del archivo PDF utilizando PyMuPDF."""
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

def preprocess_text(text):
    """Preprocesa el texto eliminando saltos de línea y dividiendo en bloques."""
    text = re.sub(r'\s+', ' ', text.strip())
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# Interfaz gráfica
root = tk.Tk()
root.title("Generador de Preguntas")
root.geometry("1200x600")  # Ajustar el tamaño para más espacio

# Frame principal
frame = tk.Frame(root, padx=10, pady=10)
frame.pack(fill="both", expand=True)

# Botones a la izquierda
tk.Button(frame, text="Subir PDFs", command=upload_and_process_pdfs).grid(row=0, column=0, padx=5, pady=5, sticky="w")
tk.Button(frame, text="Generar Pregunta", command=display_question).grid(row=1, column=0, padx=5, pady=5, sticky="w")

# Listbox a la derecha de los botones
tk.Label(frame, text="Selecciona un libro:").grid(row=0, column=1, padx=5, sticky="w")
books_listbox = tk.Listbox(frame, height=15, width=50)
books_listbox.grid(row=1, column=1, rowspan=8, padx=10, pady=5, sticky="nsew")
books_listbox.bind("<<ListboxSelect>>", select_book)

# Etiquetas a la izquierda y contenido dinámico a la derecha (Contexto, Pregunta, Opciones)
tk.Label(frame, text="Contexto:").grid(row=0, column=2, padx=5, sticky="w")
context_label = tk.Label(frame, text="", wraplength=600, justify="left", bg="#f0f0f0", relief="solid", padx=5, pady=5)
context_label.grid(row=0, column=3, padx=5, pady=5, sticky="w")

tk.Label(frame, text="Pregunta:").grid(row=1, column=2, padx=5, sticky="w")
question_label = tk.Label(frame, text="", wraplength=600, justify="left", bg="#f7f7f7", relief="solid", padx=5, pady=5)
question_label.grid(row=1, column=3, padx=5, pady=5, sticky="w")

tk.Label(frame, text="Opciones:").grid(row=2, column=2, padx=5, sticky="w")
options_label = tk.Label(frame, text="", wraplength=600, justify="left", bg="#f7f7f7", relief="solid", padx=5, pady=5)
options_label.grid(row=2, column=3, padx=5, pady=5, sticky="w")

# Cargar libros al iniciar
update_books_listbox()

# Iniciar la aplicación
root.mainloop()