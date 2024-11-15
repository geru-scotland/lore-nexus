"""
*****************************************************
 * Universidad del País Vasco (UPV/EHU)
 * Facultad de Informática - Donostia-San Sebastián
 * Asignatura: Procesamiento de Lenguaje Natural
 * Proyecto: Lore Nexus
 *
 * File: daata_normalizer.py
 * Author: geru-scotland
 * GitHub: https://github.com/geru-scotland
 * Description:
 *****************************************************
"""

import re
import unicodedata


class DataNormalizer:
    def __init__(self):
        pass

    def normalize_unicode(self, text):
        """
        """
        normalized_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        return normalized_text

    def remove_parentheses_content(self, text):
        """
        """
        return re.sub(r"\(.*?\)", "", text)

    def remove_numbers_and_punctuation(self, text):
        """
        """
        text = re.sub(r"[0-9.'’]", "", text)
        text = re.sub(r"[.-]", " ", text)
        return text

    def clean_text(self, text):
        """
        """
        text = re.sub(r'[^A-Za-z\s]', '', text)
        return text

    def normalize(self, data):
        """
        """
        if not isinstance(data, str):
            raise ValueError("Data must be a string.")

        text = self.normalize_unicode(data)
        text = self.remove_parentheses_content(text)
        text = self.remove_numbers_and_punctuation(text)
        text = self.clean_text(text)
        return text
