import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
from sklearn.preprocessing import StandardScaler

from ajustar_datos import transformador

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, '..', 'front')
STATIC_DIR = os.path.join(BASE_DIR, '..', 'front')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
CORS(app)

modelo = joblib.load('back\modelo_regresion.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    datos = request.get_json()

    valores = transformador(datos['valores'])

    scaler = StandardScaler()
    valores_escalados = scaler.fit_transform(valores)

    prediccion = modelo.predict(valores_escalados)

    return jsonify({'Costos_MillonesCOP': round(float(prediccion[0]), 3)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)