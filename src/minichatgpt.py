import nltk
import webbrowser
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator

# Descargar recursos necesarios de NLTK (solo la primera vez)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 1. Leer texto de entrenamiento
with open("entrenamiento.txt", "r", encoding="utf-8") as file:
    texto = file.read()

sentences = sent_tokenize(texto)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]

# 2. TF-IDF y vectorización
vectorizer = TfidfVectorizer()
vectorized_sentences = vectorizer.fit_transform(preprocessed_sentences)

# 3. Traductor global (se instancia una vez, no en cada loop) 
translator = Translator(service_urls=['translate.google.com'])

# 4. Funciones de utilidad 
def translate_to_english(text):
    """Traduce texto del español al inglés."""
    return translator.translate(text, src='es', dest='en').text

def translate_to_spanish(text):
    """Traduce texto del inglés al español."""
    return translator.translate(text, src='en', dest='es').text

def find_most_relevant_sentence(question):
    """Busca la oración más relevante del texto de entrenamiento."""
    preprocessed_question = preprocess_text(question)
    vectorized_question = vectorizer.transform([preprocessed_question])
    similarities = cosine_similarity(vectorized_sentences, vectorized_question)
    most_relevant_index = similarities.argmax()
    return sentences[most_relevant_index]

def msg():
    webbrowser.open("https://shorturl.at/fBEO8")

# 5. Loop principal 
print("💬 MiniGPT listo. Escribí tu pregunta (o 'salir' para terminar).")

while True:
    question = input("\n👤 Tú: ")
    if question.strip().lower() == "salir":
        print("👋 ¡Hasta luego!")
        break

    try:
        # Traducir, buscar y responder
        question_en = translate_to_english(question)
        most_relevant = find_most_relevant_sentence(question_en)
        answer_es = translate_to_spanish(most_relevant)

        msg()
        print(f"🤖 MiniGPT: {answer_es}")

    except Exception as e:
        print("⚠️ Ocurrió un error. Intenta nuevamente.")
        print("Detalles:", e)
