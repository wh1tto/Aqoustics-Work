from flask import Flask
import os
from web.routes import web_bp
from api.routes import api_bp

def create_app():
    app = Flask(__name__)
    
    app.config['SECRET_KEY'] = 'water'
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static')
    
    app.register_blueprint(web_bp, url_prefix='/')
    app.register_blueprint(api_bp, url_prefix='/api')
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
