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
import os
import random
from collections import defaultdict
from enum import Enum

from textattack.augmentation import Augmenter
from textattack.transformations import WordSwapRandomCharacterInsertion
from textattack.transformations.word_swaps.word_swap_neighboring_character_swap import WordSwapNeighboringCharacterSwap

from paths import DATA_OUTPUT_DIR, APIS_DIR, PREPROCESSING_DIR


class MythLabels(Enum):
    """
    TODO: Pasar enums a fichero estilo shared-defines
    """
    MAIN_LABEL = "Mythology"

    def __str__(self):
        return self.value

class DataProcessor:
    def __init__(self, datasets, labels, output_file, train_file, dev_file, test_file, train_pct=0.8, dev_pct=0.1, test_pct=0.1):
        self.datasets = datasets
        self.allowed_labels = self._load_labels(labels)
        self.data = self._load_data()

        self.train_pct = train_pct
        self.dev_pct = dev_pct
        self.test_pct = test_pct

        # Rutas de salida
        self.output_path = os.path.join(DATA_OUTPUT_DIR, output_file)
        self.train_path = os.path.join(DATA_OUTPUT_DIR, train_file)
        self.dev_path = os.path.join(DATA_OUTPUT_DIR, dev_file)
        self.test_path = os.path.join(DATA_OUTPUT_DIR, test_file)

    def _load_labels(self, labels):
        """
        """
        allowed_labels = []

        for label in labels:
            allowed_labels.append(f"__label__{label}")

        return allowed_labels

    def _load_data(self):
        """
        """
        filtered_data = []
        for dataset in self.datasets:
            data_dir = PREPROCESSING_DIR if dataset["path"] == "ner" else APIS_DIR
            dataset_path = os.path.join(data_dir, dataset["path"], dataset['output_folder'], dataset["output_file"])
            try:
                with open(dataset_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        if any(line.startswith(label) for label in self.allowed_labels):
                            filtered_data.append(line.strip())
            except FileNotFoundError:
                print(f"Dataset {dataset['name']} not found in path {data_dir}")
        return filtered_data

    class DataAugmentator:
        def __init__(self, outer_instance):
            self.data_processor = outer_instance

            self.swap_augmenter = Augmenter(
                transformation=WordSwapNeighboringCharacterSwap(),
                pct_words_to_swap=0.4,
                transformations_per_example=random.randint(1, 3)
            )

            self.insert_augmenter = Augmenter(
                transformation=WordSwapRandomCharacterInsertion(),
                pct_words_to_swap=0.4,
                transformations_per_example=random.randint(1, 3)
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
                    swapped_names = self.swap_augmenter.augment(name)
                    insertions_names = self.insert_augmenter.augment(name)

                    for aug_name in swapped_names + insertions_names:
                        augmented_data.append(f"{label} {aug_name}")

            return augmented_data

    def run_pipeline(self):
        augmented_data = self.augment()
        # TODO: Investigar bien esto!
        # mejores resultados sin eliminar duplicados...
        # no creo que sea bueno...
        unique_data = list(set(augmented_data))
        self._save_data(unique_data)
        self._stratify_data(unique_data)

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
        with open(self.output_path, 'r', encoding='utf-8') as file:
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
                train_size = int(self.train_pct * total_items)
                dev_size = int(self.dev_pct * total_items)

                train_lines.extend(items[:train_size])
                dev_lines.extend(items[train_size:train_size + dev_size])
                test_lines.extend(items[train_size + dev_size:])

            random.shuffle(train_lines)
            random.shuffle(dev_lines)
            random.shuffle(test_lines)

            try:
                with open(self.train_path, 'w', encoding='utf-8') as file:
                    file.writelines(train_lines)

                with open(self.dev_path, 'w', encoding='utf-8') as file:
                    file.writelines(dev_lines)

                with open(self.test_path, 'w', encoding='utf-8') as file:
                    file.writelines(test_lines)
            except FileNotFoundError:
                print("Error saving stratified data.")
