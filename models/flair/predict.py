"""
*****************************************************
 * Universidad del País Vasco (UPV/EHU)
 * Facultad de Informática - Donostia-San Sebastián
 * Asignatura: Procesamiento de Lenguaje Natural
 * Proyecto: Lore Nexus
 *
 * File: predict.py
 * Author: geru-scotland
 * GitHub: https://github.com/geru-scotland
 * Description:
 *****************************************************
"""

from flair.models import TextClassifier
from flair.data import Sentence

model_path = 'resources/taggers/universe_classifier/best-model.pt'
classifier = TextClassifier.load(model_path)

# OJO IGUAL MEJOR PASA A MINUSCULAS!!!
sentence = Sentence("Greymmane".lower())

classifier.predict(sentence, return_probabilities_for_all_classes=True)

top_labels = sorted(sentence.labels, key=lambda label: label.score, reverse=True)[:4]

for i, label in enumerate(top_labels, start=1):
    print(f"Prediction {i}: {label.value} with confidence {label.score:.4f}")
