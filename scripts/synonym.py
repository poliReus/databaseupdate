import json
import nltk
from nltk.corpus import wordnet as wn

# Scarica i dati necessari per WordNet
nltk.download('wordnet')
nltk.download('omw-1.4')

# Carica il modello Word2Vec
from gensim.models import KeyedVectors
model_path = "/Users/reus3111/sito_database/databaseupdate/combined_word2vec.vec"  # Sostituisci con il percorso del tuo file Word2Vec
model = KeyedVectors.load_word2vec_format(model_path, binary=False)

# Inizializza un dizionario per contenere i sinonimi generati da WordNet
wordnet_synonyms = {}

# Itera attraverso tutte le parole nel vocabolario
for word in model.index_to_key:
    try:
        # Trova i synset per la parola in italiano
        synsets = wn.synsets(word, lang='ita')
        # Estrai i sinonimi da ogni synset
        synonyms_set = set()
        for synset in synsets:
            synonyms_set.update(synset.lemma_names(lang='ita'))
        # Prendi solo 4 sinonimi distinti (se disponibili)
        wordnet_synonyms[word] = list(synonyms_set)[:4]
    except Exception as e:
        # Ignora errori imprevisti
        wordnet_synonyms[word] = []

# Salva il risultato in un file JSON
output_wordnet_path = "wordnet_synonyms.json"  # Sostituisci con il percorso di output desiderato
with open(output_wordnet_path, 'w', encoding='utf-8') as f:
    json.dump(wordnet_synonyms, f, ensure_ascii=False, indent=4)

print(f"File JSON generato con successo: {output_wordnet_path}")