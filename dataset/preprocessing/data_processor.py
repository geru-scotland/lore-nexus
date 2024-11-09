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
import random
from collections import defaultdict
from enum import Enum

from textattack.augmentation import Augmenter
from textattack.transformations.word_swaps.word_swap_neighboring_character_swap import WordSwapNeighboringCharacterSwap

class MythLabels(Enum):
    """
    TODO: Pasar enums a fichero estilo shared-defines
    """
    MAIN_LABEL = "Mythology"

    def __str__(self):
        return self.value

class DataProcessor:
    def __init__(self, config_path="config.json", output_path="output/dataset.txt"):
        self.config_path = config_path
        self.output_path = output_path
        self.allowed_labels = self._load_labels(config_path)
        self.datasets = self._load_datasets()
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

    def _load_datasets(self):
        """
        """
        with open(self.config_path, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)
            return config.get("datasets", [])

    def _load_data(self):
        """
        """
        filtered_data = []
        for dataset in self.datasets:
            try:
                dataset_path = dataset["path"]
                with open(dataset_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        if any(line.startswith(label) for label in self.allowed_labels):
                            filtered_data.append(line.strip())
            except FileNotFoundError:
                print(f"Dataset {dataset['name']} not found in path {dataset["path"]}")
        return filtered_data

    class DataAugmentator:
        def __init__(self, outer_instance):
            self.data_processor = outer_instance
            self.augmenter = Augmenter(
                transformation=WordSwapNeighboringCharacterSwap(),
                pct_words_to_swap=0.5,
                transformations_per_example=3
            )
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
                    name = name.strip().lower() # OJO: Estoy pasando todo a minusculas para reducir el espacio OJO EN TEST!!
                    name_parts = name.split()

                    # La original, este tiene que ir, si o si.
                    augmented_data.append(f"{label} {name}")

                    # No quiero aumentar mythology, por ahora se come a todas si no
                    if label ==  f"__label__{MythLabels.MAIN_LABEL}":
                        continue

                    # Y ahora, si el nombre tiene más de una palabra, la agrego
                    # como instancia también, CREO que puede ayudar.
                    if len(name_parts) > 1:
                        for part in name_parts:
                            new_line = f"{label} {part}"
                            augmented_data.append(new_line)

                        # OJO! igual no es buena idea pero:
                        # Asha Greyjoy -> AshaGreyjoy
                        # Que en juegos se hace muchísimo esto.
                        name_no_spaces = "".join(name_parts)
                        augmented_data.append(f"{label} {name_no_spaces}")

                    # Esto es de textattack, creo que será buena idea... veamos.
                    augmented_names = self.augmenter.augment(name)
                    for aug_name in augmented_names:
                        augmented_data.append(f"{label} {aug_name}")

            return augmented_data

    def run_pipeline(self):
        augmented_data = self.augment()
        # mejores resultados sin eliminar duplicados...
        # no creo que sea bueno...
        #unique_data = list(set(augmented_data))
        self._save_data(augmented_data)
        self._stratify_data(augmented_data)

    def augment(self):
        augmenter = self.DataAugmentator(self)
        return augmenter.augment()

    def _save_data(self, data):
        with open(self.output_path, 'w', encoding='utf-8') as file:
            for line in data:
                file.write(line + '\n')

    def _stratify_data(self, data):
        random.seed(42)

        data_by_label = defaultdict(list)
        with open('output/dataset.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # Organizo por label, para dividir equitativamente después, si no lo que me ha
            # ocurrido es que había alguna label en dev y test que no estaba en train
            for line in lines:
                label = line.split()[0]
                data_by_label[label].append(line)

            # 80% train, 10% dev/val, 10% test
            train_lines, dev_lines, test_lines = [], [], []
            for label, items in data_by_label.items():
                random.shuffle(items)

                total_items = len(items)
                train_size = int(0.88 * total_items)
                dev_size = int(0.12 * total_items)
                # test_size = total_items - train_size - dev_size

                train_lines.extend(items[:train_size])
                dev_lines.extend(items[train_size:train_size + dev_size])
                # test_lines.extend(items[train_size + dev_size:])

            random.shuffle(train_lines)
            random.shuffle(dev_lines)
            # random.shuffle(test_lines)

            try:
                with open('output/train.txt', 'w', encoding='utf-8') as file:
                    file.writelines(train_lines)

                with open('output/dev.txt', 'w', encoding='utf-8') as file:
                    file.writelines(dev_lines)

                with open('output/test.txt', 'w', encoding='utf-8') as file:
                    file.writelines(test_lines)
            except FileNotFoundError:
                print("Error saving stratified data.")



dataprocessor = DataProcessor()
dataprocessor.run_pipeline()
