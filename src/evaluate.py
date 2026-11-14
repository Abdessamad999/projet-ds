
"""Stage evaluate : métriques du modèle sur le jeu de test."""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score


# Charger le modèle entraîné
model = pickle.loads(Path("models/model.pkl").read_bytes())

# Charger les données de test
test = pd.read_csv("data/prepared/test.csv")

# Séparer les variables et la cible
y = test["diagnosis"]
pred = model.predict(test.drop(columns="diagnosis"))

# Calculer les métriques
m = {
    "accuracy": accuracy_score(y, pred),
    "f1": f1_score(y, pred),
    "recall": recall_score(y, pred),
}

# Arrondir les résultats à 4 décimales
m = {k: round(float(v), 4) for k, v in m.items()}

# Enregistrer les métriques dans metrics.json
Path("metrics.json").write_text(
    json.dumps(m, indent=2) + "\n"
)

print(m)
