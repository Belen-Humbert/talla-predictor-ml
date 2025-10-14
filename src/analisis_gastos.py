import pandas as pd
import os
from pandasai.llm.openai import OpenAI
from pandasai import PandasAI
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Obtener la API Key de OpenAI desde la variable de entorno
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "❌ OPENAI_API_KEY no encontrada. Definila en una variable de entorno o en un archivo .env"
    )

# Cargar los datos del CSV
df = pd.read_csv("../data/gastos.csv")

# Mostrar las primeras filas del dataset
print("📊 Primeras filas del dataset:")
print(df.head())

# Inicializar el modelo de lenguaje con la API Key segura
llm = OpenAI(api_token=api_key)

# Crear el asistente de análisis con PandasAI
pandas_ai = PandasAI(llm, verbose=True)

# Ejemplo de consulta a la IA:
query = "create a bar chart showing total expenses per category"

# Ejecutar la consulta
print("\n🤖 Ejecutando consulta con PandasAI...")
response = pandas_ai.run(df, query)

# Mostrar el resultado
print("\n✅ Resultado de la IA:")
print(response)
