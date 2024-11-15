"""
*****************************************************
 * Universidad del País Vasco (UPV/EHU)
 * Facultad de Informática - Donostia-San Sebastián
 * Asignatura: Procesamiento de Lenguaje Natural
 * Proyecto: Lore Nexus
 *
 * File: mythdata.py
 * Author: geru-scotland
 * GitHub: https://github.com/geru-scotland
 * Description:
 *****************************************************
"""
import re
import unicodedata
from enum import Enum

import pandas as pd

from dataset.preprocessing.data_normalizer import DataNormalizer


class MythLabels(Enum):
    """
    """
    MAIN_LABEL = "Mythology"

    def __str__(self):
        return self.value

class MythdataProcessor:
    """
        TODO: Crear clase base de la que hereden todos los "processors", como este, el de WikiData, etc.
    """
    def __init__(self, input_file, output_file):
        """
        """
        self.input_file = input_file
        self.output_file = output_file
        self.df = pd.read_csv(self.input_file)
        self.normalizer = DataNormalizer()

    def process_names(self):
        """
        """

        def normalize_and_label(text):
            """
            """
            if pd.isna(text):
                return ""

            normalized_text = self.normalizer.normalize(text)

            # A considerar si coger row['pantheon'] o no, esto es, ahora fuerzo a que sea Mythology
            label = MythLabels.MAIN_LABEL
            normalized_text = f"__label__{label} {normalized_text}"
            return normalized_text

        self.df['name'] = self.df['name'].apply(normalize_and_label)
        print("Names processing completed.")

    def save_processed_data(self):
        """
        """
        with open(self.output_file, 'w') as file:
            file.writelines(self.df['name'] + '\n')

        print(f"Processed data saved to {self.output_file}")

    def process_data(self):
        """
        """
        self.process_names()
        self.save_processed_data()
        print("Data processing completed.")
