"""
*****************************************************
 * Universidad del País Vasco (UPV/EHU)
 * Facultad de Informática - Donostia-San Sebastián
 * Asignatura: Procesamiento de Lenguaje Natural
 * Proyecto: Lore Nexus
 *
 * File: wikidata.py
 * Author: geru-scotland
 * GitHub: https://github.com/geru-scotland
 * Description:
 *****************************************************
"""
import os
from enum import Enum

import pandas as pd
import re
import unicodedata

class DatasetFormats(Enum):
    CoNLL = "CoNLL"
    FAST_TEXT = "FastText" # https://flairnlp.github.io/docs/tutorial-training/how-to-load-custom-dataset#fasttext-format
    SEPARATED_DATA_LABELS = "separated-data-labels"

    def __str__(self):
        return self.value

class WikidataProcessor:
    def __init__(self, input_file, output_file='processed_data/wikidata_dataset.csv', labels_file='labels/labels.txt'):
        """
        """

        self.input_file = input_file
        self.output_file = output_file
        self.labels_file = labels_file
        self.df = pd.read_csv(self.input_file)

    def process_labels(self):
        """
        """

        def adjust_and_homogenize_labels(text):
            universe_count = text.lower().count("universe")
            if universe_count == 1:
                text = re.sub(r"\bUniverse\b", "", text, flags=re.IGNORECASE)
            elif universe_count > 1: # Si hay más de una vez la palabra Universe, que me la quite
                text = re.sub(r"(.*)\bUniverse\b", r"\1", text, count=1, flags=re.IGNORECASE)
            text = text.strip()

            # TODO: Revisar bien las etiquetas, que tiene que haber muchas que difieran por poco
            # y hay que homogeneizar bien
            universe_mapping = {
                # "current (lower)": "homogenized_name"
                "disney": "Disney",
                "tolkien": "Tolkien",
                "star wars": "Star Wars",
                "final fantasy": "Final Fantasy",
                "ice and fire": "Game of Thrones",
                "whoniverse": "Doctor Who",
            }

            text_lower = text.lower()

            for keyword, homogeinized_name in universe_mapping.items():
                if keyword in text_lower:
                    return homogeinized_name

            return text

        self.df['universeLabel'] = self.df['universeLabel'].apply(adjust_and_homogenize_labels)
        print("Labels processing and homogenization completed.")

    def generate_label_list(self):
        """
        """

        self.process_labels()
        unique_labels = self.df['universeLabel'].unique()
        with open(self.labels_file, 'w') as file:
            for label in unique_labels:
                file.write(f"{label}\n")
        print(f"Labels written in {self.labels_file}")

    def process_names(self):
        """
        """

        def normalize_unicode(text):
            normalized_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
            normalized_text = re.sub(r"['’]", "", normalized_text)
            normalized_text = re.sub(r"[.-]", "", normalized_text)
            return normalized_text

        self.df['itemLabel'] = self.df['itemLabel'].apply(normalize_unicode)
        print("Names processing completed.")

    def process_data(self, dataset_format=None):
        """
        """

        self.process_labels()
        self.process_names()
        self.save_processed_data(dataset_format)
        print("Data processing completed.")


    def save_processed_data(self, dataset_format=None):
        """
        """

        def write_to_file(file_path, data, conll_format=False, fast_text_format=False):
            """
            """
            with open(file_path, "w") as file:
                if conll_format:
                    names = data[0]
                    labels = data[1]
                    for name, label in zip(names, labels):
                        file.write(f"{name} {label}\n")

                elif fast_text_format:
                    names, labels = data
                    for name, label in zip(names, labels):
                        label = label.replace(" ", "")
                        file.write(f"__label__{label} {name}\n")
                else:
                    for item in data:
                        file.write(f"{item}\n")

        output_file_base, extension = os.path.splitext(self.output_file)

        if dataset_format == DatasetFormats.CoNLL:
            names = self.df['itemLabel']
            labels = self.df['universeLabel']
            output_file = f"{output_file_base}_CoNLL.txt"
            write_to_file(output_file, data=[names, labels], conll_format=True)

        elif dataset_format == DatasetFormats.SEPARATED_DATA_LABELS:
            names_file = f"{output_file_base}_names.txt"
            labels_file = f"{output_file_base}_labels.txt"
            names = self.df['itemLabel']
            labels = self.df['universeLabel']
            write_to_file(names_file, data=names)
            write_to_file(labels_file, data=labels)

        elif dataset_format == DatasetFormats.FAST_TEXT:
            names = self.df['itemLabel']
            labels = self.df['universeLabel']
            output_file = f"{output_file_base}_FastText.txt"
            write_to_file(output_file, data=[names, labels], fast_text_format=True)
        else:
            self.df.to_csv(self.output_file, index=False)

        print(f"Processed data saved to {self.output_file}")


# Proceso todos los datos que he obtenido de Wikidata
processor = WikidataProcessor(input_file='raw_data/wikidata-universes.csv')
processor.process_data(dataset_format=DatasetFormats.FAST_TEXT)