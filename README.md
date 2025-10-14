# 📘 Machine Learning Intro - Aprendizaje y Práctica con Python  

Este proyecto forma parte de mi proceso de aprendizaje de **Machine Learning con Python**.  
Reúne ejercicios prácticos y experimentos personales que me ayudan a comprender mejor el análisis de datos, la predicción y el uso de inteligencia artificial aplicada.

---

## 🧠 Tecnologías y librerías usadas

- Python 3.12  
- pandas  
- scikit-learn  
- matplotlib  
- numpy  
- pandasai  
- openai  

---

## 🚀 Estructura del proyecto
machine_learning_intro/
│
├── data/
│ ├── niños.csv
│ └── gastos.csv
│
├── src/
│ ├── modelo_talla.py
│ └── analisis_gastos.py
│
├── README.md
└── requirements.txt


---

## 📊 Proyecto 1: Predicción de Talla  

El objetivo es predecir la **talla estimada** de un niño o niña según su **peso**, **altura** y **sexo**,  
utilizando un modelo de **ensamble (Voting Regressor)** que combina tres regresores diferentes.

### 🔍 Qué hace el código
1. Carga un dataset con los datos de niños/as.  
2. Divide los datos en conjuntos de entrenamiento y prueba.  
3. Aplica tres modelos de regresión:  
   - Árbol de Decisión  
   - Regresión Lineal  
   - K-Vecinos  
4. Los combina con un **Voting Regressor** (promedio de predicciones).  
5. Evalúa el rendimiento del modelo con **MSE (Error Cuadrático Medio)**.  
6. Realiza una predicción personalizada.  
7. Muestra un gráfico con los valores reales vs. predichos.

### 🎯 Propósito
Este proyecto me ayudó a comprender cómo funcionan los modelos de ensamble en Machine Learning.  
Tomar notas, comentar el código y revisar la documentación fueron pasos clave para asimilar cada concepto.

---

## 💰 Proyecto 2: Análisis de Gastos con PandasAI  

Este ejercicio está orientado al uso de **PandasAI**, una herramienta que permite analizar datos  
usando lenguaje natural. El script analiza un archivo `gastos.csv` y le pide a la IA que genere un gráfico  
con los gastos totales por categoría.

### 🧾 Qué hace el código
1. Carga un dataset con fechas, categorías y montos de gastos.  
2. Usa **PandasAI** con el modelo de OpenAI para interpretar consultas en lenguaje natural.  
3. Genera automáticamente un **gráfico de barras** con los gastos totales por categoría.  
4. Permite hacer preguntas como:
   - “Show me total expenses per day”  
   - “Create a pie chart with total expenses by category”  
   - “Which category has the highest spending?”

### 🎯 Propósito
Este segundo proyecto amplía mis conocimientos hacia el análisis de datos con IA,  
y muestra cómo automatizar tareas exploratorias usando lenguaje natural en lugar de código.

---

## ⚙️ Cómo ejecutar los proyectos

1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt

2. Ejecutar el proyecto de predicción de talla:
   ``` cd src
    python modelo_talla.py

3. Ejecutar el análisis de gastos (recordá agregar tu API Key de OpenAI):
    ```cd src
    python analisis_gastos.py

✨ Propósito general

Estoy utilizando estos proyectos como parte de mi aprendizaje práctico en Machine Learning,
análisis de datos y herramientas de IA aplicada con Python.
Cada pequeño avance me permite entender mejor los fundamentos, experimentar con nuevas librerías
y seguir fortaleciendo mi base técnica de forma autodidacta.