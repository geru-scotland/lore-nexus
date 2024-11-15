"""
*****************************************************
 * Universidad del País Vasco (UPV/EHU)
 * Facultad de Informática - Donostia-San Sebastián
 * Asignatura: Procesamiento de Lenguaje Natural
 * Proyecto: Lore Nexus
 *
 * File: slurs_extraction.py
 * Author: geru-scotland
 * GitHub: https://github.com/geru-scotland
 * Description:
 *****************************************************
"""
import os

import pandas as pd

from dataset.preprocessing.data_normalizer import DataNormalizer
from paths import APIS_DIR


class SlursProcessor:
    """
        TODO: Crear clase base de la que hereden todos los processors, como este, el de WikiData, etc.
    """
    class Config:
        """
        """
        LABEL = "__label__Offensive"
        DELIMITER = ','
        FIRST_COLUMN_NAME = "ext"

    def __init__(self, input_file, output_file, normalizer=None):
        """
        """
        self.input_file = input_file
        self.output_file = output_file
        self.normalizer = normalizer or DataNormalizer()

    def _read_dataset(self):
        """
        """
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"The file {self.input_file} doesn't exist")
        df = pd.read_csv(self.input_file, sep=self.Config.DELIMITER, usecols=[0], names=[self.Config.FIRST_COLUMN_NAME],
                         skiprows=1)
        # Mecaguen la leche, las filas que queden vacias, hay que quitar... me estaba
        # rompiendo el pipeline, porque el dataset original empieza con un par de entradas
        # numéricas y al normalizarlas, se quedaban vacías y me daba error
        df = df.dropna(subset=[self.Config.FIRST_COLUMN_NAME])
        return df

    def _write_output(self, data):
        """
        """
        with open(self.output_file, 'w', encoding='utf-8') as file:
            for _, row in data.iterrows():
                slur = self.normalizer.normalize(row[self.Config.FIRST_COLUMN_NAME]).strip()
                if slur:
                    file.write(f"{self.Config.LABEL} {slur}\n")

    def process(self):
        """
        """
        print(f"Reading dataset from {self.input_file}...")
        data = self._read_dataset()
        print(f"PRocessing {len(data)} lines...")
        self._write_output(data)
        print(f"Labeled file: {self.output_file}")
