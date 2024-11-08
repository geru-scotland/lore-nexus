"""
*****************************************************
 * Universidad del País Vasco (UPV/EHU)
 * Facultad de Informática - Donostia-San Sebastián
 * Asignatura: Procesamiento de Lenguaje Natural
 * Proyecto: Lore Nexus
 *
 * File: data_processor.py
 * Author: geru-scotland
 * GitHub: https://github.com/geru-scotland
 * Description:
 *****************************************************
"""
import json

class DataProcessor:
    def __init__(self, input_path="apis/wikidata/processed_data/wikidata_dataset_FastText.txt", output_path="output/dataset.txt", config_path="config.json"):
        self.input_path = input_path
        self.output_path = output_path
        self.allowed_labels = self._load_labels(config_path)
        self.data = self._load_data()

    def _load_labels(self, config_path):
        """
        """
        allowed_labels = []
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)

            raw_labels = config.get("labels", [])

            for label in raw_labels:
                allowed_labels.append(f"__label__{label}")

        return allowed_labels

    def _load_data(self):
        """
        """
        filtered_data = []
        with open(self.input_path, 'r', encoding='utf-8') as file:
            for line in file:
                if any(line.startswith(label) for label in self.allowed_labels):
                    filtered_data.append(line.strip())
        return filtered_data

    class DataAugmentator:
        def __init__(self, outer_instance):
            self.data_processor = outer_instance

        def augment(self):
            """

            TODO: Aumentar también con variaciones de nombres de cada universo, por ejemplo:
            - Star Wars: Para Chewbacca, Chebaka, Chewaka, Chewbaka
            - Final Fantasy: Para Cloud, Cloudd, Cloudo, Claud, Clouud, Chloud
            """
            augmented_data = []
            for line in self.data_processor.data:
                if line.startswith("__label__"):
                    label, name = line.split(" ", 1)
                    name_parts = name.strip().split()

                    # Multiples palabras por nombre, este tiene que ir, si o si.
                    augmented_data.append(line.strip())

                    # Y ahora, si el nombre tiene más de una palabra, la agrego
                    # como instancia también, CREO que puede ayudar.
                    if len(name_parts) > 1:
                        for part in name_parts:
                            new_line = f"{label} {part}"
                            augmented_data.append(new_line)

            return augmented_data

    def run_pipeline(self):
        augmented_data = self.augment()
        # mejores resultados sin eliminar duplicados...
        # no creo que sea bueno...
        #unique_data = list(set(augmented_data))
        self._save_data(augmented_data)

    def augment(self):
        augmenter = self.DataAugmentator(self)
        return augmenter.augment()

    def _save_data(self, data):
        with open(self.output_path, 'w', encoding='utf-8') as file:
            for line in data:
                file.write(line + '\n')


dataprocessor = DataProcessor()
dataprocessor.run_pipeline()
