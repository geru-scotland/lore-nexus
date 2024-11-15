"""
*****************************************************
 * Universidad del País Vasco (UPV/EHU)
 * Facultad de Informática - Donostia-San Sebastián
 * Asignatura: Procesamiento de Lenguaje Natural
 * Proyecto: Lore Nexus
 *
 * File: ner_corpus_builder.py
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

from dataset.preprocessing.data_normalizer import DataNormalizer


class PATHS(Enum):
    BOOKS = "books"
    OUTPUT = "extracted_entities"

    # para que se pueda imprimir el valor del enum
    # sin tener que andar haciendo .value
    def __str__(self):
        return self.value


class EntityCorpusBuilder:
    class Extractor:
        def __init__(self, nlp):
            self.nlp = nlp
            self.normalizer = DataNormalizer()

        def extract_entities(self, pdf_path):
            entities = set() # diccionario, para no repetir entidades
            max_length = 100000  # tamaño máximo de chunk a procesar, que casca si es muy grande

            text = extract_text(pdf_path)

            if text:
                text_chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
                for chunk in text_chunks:
                    doc = self.nlp(chunk)
                    for entity in doc.ents:
                        if entity.label_ in ["LOC", "GPE"]:
                            # solo letras
                            # TODO: utilizar unicodedata para normalizar mejor
                            clean_text = self.normalizer.normalize().replace(" ", "")
                            if clean_text != entity.text:
                                clean_text = clean_text.replace(" ", "")
                            if clean_text and len(clean_text) > 3:
                                entities.add(clean_text)
            return entities

        def process_all_pdfs_in_folder(self):
            folder = Path(str(PATHS.BOOKS))
            output_folder = Path(str(PATHS.OUTPUT))

            # Cojo todos los pdfs
            for pdf_file in folder.glob("*.pdf"):
                output_file = output_folder / f"{pdf_file.stem}_entities.txt"

                # Compruebo que no exista el output, mismo nombre que
                # pdf pero terminante en _entities.txt
                if not output_file.exists():
                    print(f"Extracting entities from: {pdf_file.name}")
                    entities = self.extract_entities(pdf_file)
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_file, "w") as file:
                        for entity in entities:
                            file.write(f"{entity}\n")
                    print(f"Entities from {pdf_file.name} saved to {output_file}")
                else:
                    print(f"Already processed, skipping: {pdf_file.name}")

    def __init__(self, model="en_core_web_trf"):
        # modelo pre-entrenado de spaCy, el de "es_core_news_sm" no iba muy bien
        # Este es una bestia, utiliza RoBERTa
        self.nlp = spacy.load(model)
        self.nlp.max_length = 2000000
        self.extractor = self.Extractor(self.nlp)

    def build(self, book_name=None):
        if book_name:
            bookpath = f"{PATHS.BOOKS}/{book_name}"
            output_file = f"{PATHS.OUTPUT}/{book_name.split('.')[0]}_entities.txt"
            print(f"Extracting entities from: {bookpath}")
            try:
                entities = self.extractor.extract_entities(bookpath)
                with open(output_file, "w") as file:
                    for entity in entities:
                        file.write(f"{entity}\n")
                print(f"Entities extracted to {output_file}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            self.extractor.process_all_pdfs_in_folder()

    def label_and_filter_entities(self, input_dir="raw_data", output_file="processed_data/ner_dataset.txt"):
        input_path = Path(input_dir)
        output_path = Path(output_file)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        label_map = {
            "got": "__label__GameofThrones",
            "lotr": "__label__Tolkien",
            "sw": "__label__StarWars",
            "wow-wotlk": "__label__Warcraft",
            "hp": "__label__HarryPotter"
        }

        with open(output_path, "w") as f_out:
            for file in input_path.glob("*.txt"):
                match = re.match(r"([a-zA-Z\-]+)", file.stem)
                if match:
                    file_prefix = match.group(1)
                    label = label_map.get(file_prefix, None)

                    if label:
                        with open(file, "r") as f_in:
                            for line in f_in:
                                entity = line.strip()

                                if len(entity) > 3:
                                    f_out.write(f"{label} {entity}\n")
                        print(f"Processed file: {file.name}")
                    else:
                        print(f"No label found for file: {file.name}, skipping.")

corpus_builder = EntityCorpusBuilder()
corpus_builder.label_and_filter_entities()




