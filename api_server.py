"""API server example"""

#
# Usage from command line:
# Desde bash: curl http://127.0.0.1:5000 -X POST -H "Content-Type: application/json" -d '{"bathrooms": "2", "bedrooms": "3", "sqft_living": "1800", "sqft_lot": "2200", "floors": "1", "waterfront": "1", "condition": "3"}'
# Desde PowerShell: Invoke-RestMethod -Uri "http://127.0.0.1:5000" -Method POST -Headers @{ "Content-Type" = "application/json" } -Body '{ "bathrooms": "2", "bedrooms": "3", "sqft_living": "1800", "sqft_lot": "2200", "floors": "1", "waterfront": "1", "condition": "3" }'

import pickle
import pandas as pd
from flask import Flask, request

'''Flask es un paquete de python que permite crear aplicaciones de tipo
 cliente - servidor (flask es el servidor)'''


app = Flask(__name__)
app.config["SECRET_KEY"] = "you-will-never-guess"


FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]


# Ejecuta la función index cuando el servidor recibe una petición POST
@app.route("/", methods=["POST"])
def index():
    """API function"""

    # request.json es un diccionario con los datos que se envían en la petición
    args = request.json
    filt_args = {key: [int(args[key])] for key in FEATURES}
    df = pd.DataFrame.from_dict(filt_args)

    # Cargar el modelo
    with open("house_predictor.pickle", "rb") as file:
        loaded_model = pickle.load(file)

    # Predecir el precio con los valores enviados
    prediction = loaded_model.predict(df)

    return str(prediction[0][0])


if __name__ == "__main__":
    app.run(debug=True)