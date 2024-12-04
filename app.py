import os
from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import json
import re

app = Flask(__name__)

# Directorio para guardar los archivos PDF temporalmente
UPLOAD_FOLDER = './uploads'
JSON_FOLDER = './json_outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['JSON_FOLDER'] = JSON_FOLDER

# Asegurarse de que el directorio de subida exista
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(JSON_FOLDER):
    os.makedirs(JSON_FOLDER)


# Ruta para cargar múltiples archivos PDF
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part", "details": "No se encontró el campo 'file' en la solicitud."}), 400

    files = request.files.getlist('file')
    if len(files) == 0:
        return jsonify({"error": "No files selected", "details": "No se seleccionaron archivos."}), 400

    processed_files = []
    json_data = {}
    
    for file in files:
        if file.filename == '':
            return jsonify({"error": "One or more files have no filename", "details": "Al menos un archivo no tiene nombre."}), 400
        
        if file and allowed_file(file.filename):
            # Guardar cada archivo temporalmente
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

        
            # Extraer el texto de cada archivo PDF
            text = extract_text_from_pdf(file_path)


            # Preprocesar el textos
            preprocessed_text = preprocess_text(text)
            processed_files.append(file.filename)  # Agregar los primeros 500 caracteres del texto

            # Agregar el texto al diccionario para generar el JSON
            json_data[file.filename] = preprocessed_text
            
        else:
            return jsonify({"error": "Invalid file type", "details": f"El archivo {file.filename} no es un PDF válido."}), 400
        
        # Guardar el diccionario como un archivo JSON
        json_output_path = os.path.join(app.config['JSON_FOLDER'], 'processed_data.json')
        with open(json_output_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, ensure_ascii=False, indent=4)

    # Responder con los archivos procesados y el texto extraído
    return jsonify({
        "message": "PDFs processed successfully",
        "files": processed_files,
        "json_path": json_output_path
    })

def allowed_file(filename):
    """ Verifica si el archivo tiene una extensión permitida. """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['pdf']

def extract_text_from_pdf(file_path):
    """ Extrae el texto del archivo PDF utilizando PyMuPDF """
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")  # Extrae el texto de la página
    return text

def preprocess_text(text):
    """
    Preprocesa el texto eliminando saltos de línea y dividiendo en bloques
    según signos de puntuación.
    """
    # Eliminar saltos de línea y exceso de espacios
    text = re.sub(r'\s+', ' ', text.strip())

    # Dividir texto en bloques según los signos de puntuación
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Filtrar bloques vacíos o insignificantes
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    return sentences

if __name__ == '__main__':
    app.run(debug=True)
