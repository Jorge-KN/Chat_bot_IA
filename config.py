# config.py

class Config:
    # No se usa SECRET_KEY
    DEBUG = True  # Activar el modo de depuración
    TESTING = False  # No activamos el modo de pruebas
    # Si es necesario, puedes definir otras configuraciones, como base de datos, etc.
    # Ejemplo:
    # SQLALCHEMY_DATABASE_URI = 'mysql://user:password@localhost/dbname'
    # SQLALCHEMY_TRACK_MODIFICATIONS = False