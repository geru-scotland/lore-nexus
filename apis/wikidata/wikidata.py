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
import pandas as pd
import re
import unicodedata


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
                "disney": "Disney",
                "tolkien": "Tolkien",
                "star wars": "Star Wars",
                "final fantasy": "Final Fantasy",
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
            return normalized_text

        self.df['itemLabel'] = self.df['itemLabel'].apply(normalize_unicode)
        print("Names processing completed.")

    def process_data(self):
        """
        """
        self.process_labels()
        self.process_names()
        self.save_processed_data()
        print("Data processing completed.")


    def save_processed_data(self):
        """
        """
        self.df.to_csv(self.output_file, index=False)
        print(f"Processed data saved to {self.output_file}")


# Proceso todos los datos que he obtenido de Wikidata
processor = WikidataProcessor(input_file='raw_data/wikidata-universes.csv')
processor.process_data()