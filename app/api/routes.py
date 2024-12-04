from flask import Blueprint, request, jsonify
from app.api.models import load_data, generate_question

# Crear un Blueprint para las rutas de la API
api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/load_data', methods=['POST'])
def load_data_route():
    # Lógica para cargar datos desde un archivo (JSON, por ejemplo)
    data = request.get_json()
    result = load_data(data)
    return jsonify(result)

@api_blueprint.route('/generate_question', methods=['POST'])
def generate_question_route():
    # Lógica para generar preguntas usando el modelo T5
    data = request.get_json()
    question = generate_question(data['content'])
    return jsonify({'question': question})
