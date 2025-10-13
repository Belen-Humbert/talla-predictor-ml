# 📘 Machine Learning Intro - Predicción de Talla

Este proyecto es parte de mi proceso de aprendizaje de **Machine Learning con Python**.  
El objetivo es predecir la **talla estimada** de un niño o niña en función de su **peso**, **altura** y **sexo**, utilizando un modelo de **ensamble (Voting Regressor)**.

---

## 🧠 Tecnologías y librerías usadas

- Python 3.12
- pandas
- scikit-learn
- matplotlib
- numpy

---

## 🚀 Estructura del proyecto
machine_learning_intro/
│
├── data/
│ └── niños.csv
│
├── src/
│ └── modelo_talla.py
│
├── README.md
└── requirements.txt


---

## 🧩 Cómo ejecutar

1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt

2. Ejecutar el script:
cd src
python modelo_talla.py

## 📊 Qué hace el código
1. Carga un dataset con los datos de niños/as.
2. Divide el dataset en entrenamiento y prueba.
3. Aplica tres modelos de regresión:
    Árbol de decisión
    Regresión lineal
    K-vecinos

4. Los combina con un Voting Regressor (promedio ponderado de predicciones).
5. Evalúa el modelo con MSE.
6. Realiza una predicción personalizada.
7. Muestra un gráfico con los valores reales vs predichos.

## ✨ Propósito
Estoy usando estos ejemplos para afianzar mis bases en Machine Learning aplicado con Python y Scikit-Learn.