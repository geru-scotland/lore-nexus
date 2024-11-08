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
from pathlib import Path

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
        entities = set()
        max_length = 100000  # tamaño máximo de chunk a procesar, que casca si es muy grande

        text = extract_text(pdf_path)

        if text:
            text_chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]

            for chunk in text_chunks:
                doc = self.nlp(chunk)
                for entity in doc.ents:
                    if entity.label_ in ["PERSON", "LOC", "GPE", "ORG", "NORP", "FAC", "WORK_OF_ART", "EVENT"]:
                        # solo letras
                        # TODO: utilizar unicodedata para normalizar mejor
                        clean_text = re.sub(r'[^A-Za-z\s]', '', entity.text)
                        # Si se ha quitado un carácter, que me quite los espacios
                        # (suelen tener muchos ', ó ^ nombres de elfos etc)
                        if clean_text != entity.text:
                            clean_text = clean_text.replace(" ", "")
                        if clean_text:
                            entities.add(clean_text)
        return entities

    def extract_and_export(self, book_name):
        bookpath = f"{PATHS.BOOKS}/{book_name}"
        output_file = f"{PATHS.OUTPUT}/{book_name.split(".")[0]}_entities.txt"

        try:
            entities = self.extract_entities(bookpath)
            with open(output_file, "w") as file:
                file.write("\n".join(entities))
            print(f"Entities extracted to {output_file}")
        except Exception as e:
            print(f"Error: {e}")

    def process_all_pdfs_in_folder(self):
        folder = Path(str(PATHS.BOOKS))
        output_folder = Path(str(PATHS.OUTPUT))

        # Cojo todos los pdfs
        for pdf_file in folder.glob("*.pdf"):
            output_file = output_folder / f"{pdf_file.stem}_entities.txt"

            # Compruebo que no exista el output, mismo nombre que
            # pdf pero terminante en _entities.txt
            if not output_file.exists():
                print(f"Processing: {pdf_file.name}")
                entities = self.extract_entities(pdf_file)

                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w") as file:
                    for entity in entities:
                        file.write(f"{entity}\n")
                print(f"Entities from {pdf_file.name} saved to {output_file}")
            else:
                print(f"Already processed, skipping: {pdf_file.name}")


ner_extractor = NERextractor()
ner_extractor.process_all_pdfs_in_folder()




