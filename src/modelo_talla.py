import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Modelos base
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# Preprocesamiento y métricas
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

# 1️⃣ Cargar los datos
data = pd.read_csv('../data/niños.csv')

# 2️⃣ Separar las características (X) y la etiqueta (y)
X = data[['peso', 'sexo', 'altura']]
y = data['talla']

# 3️⃣ Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Definir los modelos base
models = [
    ('decision_tree', DecisionTreeRegressor(random_state=42)),
    ('linear_regression', LinearRegression()),
    ('k_neighbors', KNeighborsRegressor(n_neighbors=5))
]

# 5️⃣ Escalar los datos y aplicar el ensamble (Voting Regressor)
#    Esto crea un "pipeline" que primero normaliza los datos y luego entrena el modelo
scaler = StandardScaler()
ensemble = VotingRegressor(models)
model = Pipeline([
    ('scaler', scaler),
    ('ensemble', ensemble)
])

# 6️⃣ Entrenar el modelo
model.fit(X_train, y_train)

# 7️⃣ Evaluar el modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE (error cuadrático medio) en test: {mse:.3f}")

# 8️⃣ Nueva predicción
peso = 50
sexo = 0  # 0: masculino, 1: femenino
altura = 158
nueva_muestra = pd.DataFrame([[peso, sexo, altura]], columns=['peso', 'sexo', 'altura'])

talla_predicha = model.predict(nueva_muestra)[0]

print(f"\nPredicción: para un peso de {peso} kg, altura de {altura} cm y sexo masculino,")
print(f"la talla estimada es {talla_predicha:.2f}\n")

# 9️⃣ Visualización: valores reales vs predichos
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.title('Comparación entre valores reales y predichos')
plt.xlabel('Talla real')
plt.ylabel('Talla predicha')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
