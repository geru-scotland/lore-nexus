from enum import Enum

import pdfplumber
import spacy

class PATHS(Enum):
    BOOKS = "books"
    OUTPUT = "extracted_entities"

    # para que se pueda imprimir el valor del enum
    # sin tener que andar haciendo .value
    def __str__(self):
        return self.value

class NERextractor:
    # modelo pre-entrenado de spaCy
    def __init__(self, model="es_core_news_sm"):
        self.nlp = spacy.load(model)

    def extract_entities(self, pdf_path):
        # diccionario, para no repetir entidades
        entities = {}
        with pdfplumber.open(pdf_path) as pdf:
            for page in range(len(pdf.pages)):
                text = pdf.pages[page].extract_text()
                doc = self.nlp(text)
                # Quizá extraer solo PER y LOC
                for entity in doc.ents:
                    if entity.text not in entities:
                        entities[entity.text] = entity.label_
        return entities

    def extract_and_export(self, book_name):
        # TODO: hacer que lea todos los pdf de book folder
        bookpath = f"{PATHS.BOOKS}/{book_name}"
        output_file = f"{PATHS.OUTPUT}/{book_name}_entities.txt"
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




