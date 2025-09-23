import spacy

# Carica il modello per l'italiano
nlp = spacy.load("it_core_news_sm")

testo = "Il gatto nero salta agilmente sul muro alto."

print(f"\nAnalisi del testo: {testo}\n{'-'*50}")
doc = nlp(testo)
# Output analisi
for token in doc:
    print(f"Token: {token.text:12} | Lemma: {token.lemma_:12} | POS: {token.pos_:10} | Dipendenza: {token.dep_:10} | Head: {token.head.text}")
