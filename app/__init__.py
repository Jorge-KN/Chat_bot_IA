from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # Cargar configuraciones
    app.config.from_object('app.config.Config')

    # Importar y registrar las rutas
    from app.api.routes import api_blueprint
    app.register_blueprint(api_blueprint)

    return app
