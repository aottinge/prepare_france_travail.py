import cohere
import numpy as np
import pandas as pd
import requests

# 1. Charger vos données de test
with open("mes_textes.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]

print(f"Lignes chargées : {len(texts)}")

# 2. API Cohere (1024 dimensions)
co = cohere.Client("VOTRE_CLE_API_COHERE")
cohere_resp = co.embed(
    texts=texts, model="embed-multilingual-v3.0", input_type="search_document"
)
# Extraction propre des vecteurs pour Cohere v3
cohere_vectors = cohere_resp.embeddings

# 3. API Jina AI v4 (Tronqué à 1024 dimensions via Matryoshka + Task Spécifique)
headers = {
    "Authorization": "Bearer VOTRE_CLE_API_JINA",
    "Content-Type": "application/json",  # Indispensable pour l'API Jina
}
json_data = {
    "model": "jina-embeddings-v4",
    "dimensions": 1024,
    "task": "retrieval.passage",  # Obligatoire pour Jina v4 (optimise l'indexation de documents)
    "input": texts,
}

response = requests.post(
    "https://api.jina.ai/v1/embeddings", headers=headers, json=json_data
)

# Gestion d'erreur au cas où l'API Jina renvoie une erreur (ex: clé invalide)
if response.status_code != 200:
    print(f"Erreur Jina API : {response.text}")
    exit()

jina_resp = response.json()
jina_vectors = [item["embedding"] for item in jina_resp["data"]]

# 4. Sauvegarder au format attendu par embcompare
# Fichier TSV pour Cohere
df_cohere = pd.DataFrame({"label": texts})
df_vectors_cohere = pd.DataFrame(cohere_vectors)
df_cohere = pd.concat([df_cohere, df_vectors_cohere], axis=1)
df_cohere.to_csv("vectors_cohere.tsv", sep="\t", index=False)

# Fichier TSV pour Jina
df_jina = pd.DataFrame({"label": texts})
df_vectors_jina = pd.DataFrame(jina_vectors)
df_jina = pd.concat([df_jina, df_vectors_jina], axis=1)
df_jina.to_csv("vectors_jina.tsv", sep="\t", index=False)

print("Fichiers 'vectors_cohere.tsv' et 'vectors_jina.tsv' prêts !")