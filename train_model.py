"""Build, deploy and access a model using scikit-learn"""

''' Faltaría probar con distintos modelos, analizar los datos
 y ver cuál es el que mejor se ajusta a los datos.'''

import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("house_data.csv", sep=",")

# Extrae las columnas de interés para el problema (predictoras)
features = df[
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "condition",
    ]
]

# Extrae la columna objetivo, la que se quiere predecir (respuesta)
target = df[["price"]]

# Ajusta el modelo de regresión lineal a los datos (un objeto en scikit-learn)
estimator = LinearRegression()

# Calcula los coeficientes del modelo
estimator.fit(features, target)

# Coeficientes del modelo
print(estimator.coef_)
print(estimator.intercept_)

# Abrimos el archivo como escritura binaria
with open("house_predictor.pickle", "wb") as file:
    # Guardamos el modelo (objeto) en el archivo, para poder usarlo después
    pickle.dump(estimator, file)
    