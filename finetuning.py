import json
import numpy as np
from gensim.models import KeyedVectors

# Carica il modello Word2Vec esistente
model_path = "./combined_word2vec.vec"
model = KeyedVectors.load_word2vec_format(model_path, binary=False)

# Copia mutabile dei vettori
mutable_vectors = {word: np.copy(model[word]) for word in model.index_to_key}
vector_size = model.vector_size

# Carica il feedback dal file JSON
with open("feedback.json", "r") as f:
    feedback_data = json.load(f)

# Preprocessamento semplice per il testo
def preprocess_text(text):
    return text.lower().split()

# Funzione per aggiungere parole mancanti
def add_word_to_vectors(word, vectors, vector_size):
    if word not in vectors:
        vectors[word] = np.random.normal(size=(vector_size,))

# Funzione per aggiornare i vettori con fine-tuning
def fine_tune_vectors_with_feedback(vectors, feedback_data, learning_rate=0.01):
    for entry in feedback_data:
        query = preprocess_text(entry["query"])
        giocatore_data = preprocess_text(" ".join(
            [str(value) for key, value in entry["giocatore"].items() if value and isinstance(value, str)]
        ))
        label = entry["label"]

        # Aggiungi parole mancanti
        for word in query + giocatore_data:
            add_word_to_vectors(word, vectors, vector_size)

        # Media dei vettori della query e dei dati del giocatore
        query_vector = np.mean([vectors[word] for word in query if word in vectors], axis=0)
        giocatore_vector = np.mean([vectors[word] for word in giocatore_data if word in vectors], axis=0)

        # Aggiorna i vettori delle parole in base al feedback
        for word in query + giocatore_data:
            if word in vectors:
                adjustment = learning_rate * (query_vector - giocatore_vector if label == 0 else giocatore_vector - query_vector)
                vectors[word] += adjustment

    return vectors

# Esegui il fine-tuning
mutable_vectors = fine_tune_vectors_with_feedback(mutable_vectors, feedback_data)

# Ricrea il modello Word2Vec aggiornato
updated_model = KeyedVectors(vector_size=vector_size)
for word, vector in mutable_vectors.items():
    updated_model.add_vector(word, vector)

# Salva il modello aggiornato
updated_model_path = "./updated_word2vec_with_fine_tuning.vec"
updated_model.save_word2vec_format(updated_model_path, binary=False)

print(f"Modello aggiornato con fine-tuning salvato in: {updated_model_path}")