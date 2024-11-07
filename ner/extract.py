"""
*****************************************************
 * Universidad del País Vasco (UPV/EHU)
 * Facultad de Informática - Donostia-San Sebastián
 * Asignatura: Procesamiento de Lenguaje Natural
 * Proyecto: Lore Nexus
 *
 * File: extract.py
 * Author: geru-scotland
 * GitHub: https://github.com/geru-scotland
 * Description:
 *****************************************************
"""
from enum import Enum
import re

from pdfminer.high_level import extract_text
import spacy

class PATHS(Enum):
    BOOKS = "books"
    OUTPUT = "extracted_entities"

    # para que se pueda imprimir el valor del enum
    # sin tener que andar haciendo .value
    def __str__(self):
        return self.value

class NERextractor:
    # modelo pre-entrenado de spaCy, el de "es_core_news_sm" no iba muy bien
    # Este es una bestia, utiliza RoBERTa
    def __init__(self, model="en_core_web_trf"):
        self.nlp = spacy.load(model)
        self.nlp.max_length = 2000000

    def extract_entities(self, pdf_path):
        # diccionario, para no repetir entidades
        entities = {}
        text = extract_text(pdf_path)

        if text:
            doc = self.nlp(text)
            for entity in doc.ents:
                # al final cojo solo LOC y PER
                # TODO: Cuando no registres etiquetas, cambiar de dict a set
                if entity.label_ in ["PERSON", "LOC", "GPE", "ORG", "NORP", "FAC", "WORK_OF_ART", "EVENT"]:
                    # solo letras
                    clean_text = re.sub(r'[^A-Za-z\s]', '', entity.text)
                    # Si se ha quitado un carácter, que me quite los espacios
                    # (suelen tener muchos ', ó ^ nombres de elfos etc)
                    if clean_text != entity.text:
                        clean_text = clean_text.replace(" ", "")
                    if clean_text:
                        entities[clean_text] = entity.label_
        return entities

    def extract_and_export(self, book_name):
        # TODO: hacer que lea todos los pdf de book folder
        bookpath = f"{PATHS.BOOKS}/{book_name}"
        output_file = f"{PATHS.OUTPUT}/{book_name.split(".")[0]}_entities.txt"
        try:
            entities = self.extract_entities(bookpath)
            with open(output_file, "w") as file:
                for entity, label in entities.items():
                    file.write(f"{entity}: {label}\n")
            print(f"Entities extracted to {output_file}")
        except Exception as e:
            print(f"Error: {e}")


ner_extractor = NERextractor()
ner_extractor.extract_and_export("lotr1.pdf")




