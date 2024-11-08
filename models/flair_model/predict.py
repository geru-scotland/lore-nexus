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

sentence = Sentence("Chewbawka")

classifier.predict(sentence)

print(f"Predicted label: {sentence.labels[0].value}")
print(f"Confidence: {sentence.labels[0].score:.4f}")
