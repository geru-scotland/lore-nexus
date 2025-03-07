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

from dataset.preprocessing.data_normalizer import DataNormalizer
from paths import APIS_DIR


class DatasetFormats(Enum):
    CoNLL = "CoNLL"
    FAST_TEXT = "FastText"  # https://flairnlp.github.io/docs/tutorial-training/how-to-load-custom-dataset#fasttext-format
    SEPARATED_DATA_LABELS = "separated-data-labels"

    def __str__(self):
        return self.value


class WikidataProcessor:
    """
        TODO: Crear clase base de la que hereden todos los processors , como este, el de MythData etc, etc.
    """

    def __init__(self, input_file, output_file, labels_file, historical_file):
        """
        """

        self.input_file = input_file
        self.output_file = output_file
        self.labels_file = labels_file
        self.historical_file = historical_file

        self.df = pd.read_csv(self.input_file)
        self.normalizer = DataNormalizer()

    def process_labels(self):
        """
        """

        def adjust_and_homogenize_labels(text):
            universe_count = text.lower().count("universe")
            if universe_count == 1:
                text = re.sub(r"\bUniverse\b", "", text, flags=re.IGNORECASE)
            elif universe_count > 1:  # Si hay más de una vez la palabra Universe, que me la quite
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
        Simplemente para generar un fichero con todas las etiquetas únicas
        Por curiosidad, no forma parte del proceso de pipeline.
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

        self.df['itemLabel'] = self.df['itemLabel'].apply(self.normalizer.normalize)
        print("Names processing completed.")

    def label_append_historical_data(self):
        """
        """
        data = []
        with open(self.historical_file, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    data.append(self.normalizer.normalize(line))

        historical_df = pd.DataFrame({
            "universeLabel": ["Historical"] * len(data),
            "itemLabel": data
        })

        self.df = pd.concat([self.df, historical_df], ignore_index=True)
        print("Historical data appended.")

    def process_data(self, dataset_format=None):
        """
        """
        # TODO: Incluir el fichero de personajes históricos, simplemente hacer un append
        self.process_labels()
        self.process_names()
        self.label_append_historical_data()
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

        if dataset_format == DatasetFormats.CoNLL:
            names = self.df['itemLabel']
            labels = self.df['universeLabel']
            write_to_file(self.output_file, data=[names, labels], conll_format=True)

        elif dataset_format == DatasetFormats.SEPARATED_DATA_LABELS:
            names_file = os.path.join(APIS_DIR, "wikidata", "labels", "wikidata_dataset_names.txt")
            names = self.df['itemLabel']
            labels = self.df['universeLabel']
            write_to_file(names_file, data=names)
            write_to_file(self.labels_file, data=labels)

        elif dataset_format == DatasetFormats.FAST_TEXT:
            names = self.df['itemLabel']
            labels = self.df['universeLabel']
            write_to_file(self.output_file, data=[names, labels], fast_text_format=True)
        else:
            output_file = os.path.join(APIS_DIR, "wikidata", "wikidata_dataset.csv")
            self.df.to_csv(output_file, index=False)

        print(f"Processed data saved to {self.output_file}")
