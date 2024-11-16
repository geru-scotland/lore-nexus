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
from textattack.transformations import (
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterDeletion
)

from paths import DATA_OUTPUT_DIR, APIS_DIR, PREPROCESSING_DIR


class MythLabels(Enum):
    """
    TODO: Pasar enums a fichero estilo shared-defines
    """
    MAIN_LABEL = "Mythology"

    def __str__(self):
        return self.value

class DataProcessor:
    def __init__(self, datasets, labels, unique_names, augmentation, output_file, train_file, dev_file, test_file, train_pct=0.8, dev_pct=0.1, test_pct=0.1):
        self.datasets = datasets
        self.allowed_labels = self._load_labels(labels)
        self.unique_names = unique_names
        self.augmentation_config = augmentation
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
        class AugmentationConfig:
            def __init__(self, config):

                self.config = config

                self.swap_augmenter = self._create_augmenter(
                    "swap_characters", WordSwapNeighboringCharacterSwap()
                )
                self.insert_augmenter = self._create_augmenter(
                    "insert_characters", WordSwapRandomCharacterInsertion()
                )
                self.delete_augmenter = self._create_augmenter(
                    "delete_characters", WordSwapRandomCharacterDeletion()
                )
                self.duplicate_char_augmenter = self._create_augmenter(
                    "duplicate_characters", WordSwapRandomCharacterInsertion()
                )

                self.internal_swap_enabled = config["internal_swap"]["enabled"]
                self.internal_swap_probability = config["internal_swap"]["swap_probability"]
                self.split_names_enabled = config["split_names"]["enabled"]
                self.join_parts = config["split_names"].get("join_parts", False)
                self.label_exclusion_enabled = config["label_exclusion"]["enabled"]
                self.excluded_labels = config["label_exclusion"].get("excluded_labels", [])

            def _create_augmenter(self, key, transformation):
                if self.config[key]["enabled"]:
                    transform_num = 0
                    if key == "duplicate_characters":
                        transform_num = self.config[key]["transformations_per_example"]
                    else:
                        transform_num = random.randint(
                        self.config[key]["transformations_per_example"]["min"],
                        self.config[key]["transformations_per_example"]["max"]
                        )
                    return Augmenter(
                        transformation=transformation,
                        pct_words_to_swap=self.config[key]["pct_words_to_swap"],
                        transformations_per_example=transform_num
                    )
                return None

        def __init__(self, outer_instance):
            self.data_processor = outer_instance
            self.config = self.AugmentationConfig(outer_instance.augmentation_config)

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

                    # La original, este tiene que ir, si o si.
                    augmented_data.append(f"{label} {name}")

                    # Coloco aquí para que las que están excluidas, al menos se
                    # cree una instancia para aquellas que son palabras compuestas
                    if self.config.join_parts:
                        # OJO! igual no es buena idea pero:
                        # Asha Greyjoy -> AshaGreyjoy
                        # Que en juegos se hace muchísimo esto.
                        name_no_spaces = "".join(name.split())
                        augmented_data.append(f"{label} {name_no_spaces}")

                    # Modifico esto para exluir más lables, desde el config fácilmente
                    if self.config.label_exclusion_enabled:
                        excluded_labels = []
                        for excluded_label in self.config.excluded_labels:
                            excluded_labels.append(f"__label__{excluded_label}")

                        if label in excluded_labels:
                            continue

                    # swap de chars internos, sin tocar el primero y último
                    if self.config.internal_swap_enabled:
                        internal_swap_name = self._internal_swap(name)
                        augmented_data.append(f"{label} {internal_swap_name}")

                    # duplico aleatoriamente un char, para casos como Jon Snow -> Jonn Snow
                    if self.config.duplicate_char_augmenter:
                        duplicated_char_names = self.config.duplicate_char_augmenter.augment(name)
                        augmented_data.extend([f"{label} {dup_name}" for dup_name in duplicated_char_names])


                    # Y ahora, si el nombre tiene más de una palabra, la agrego
                    # como instancia también, CREO que puede ayudar.
                    if self.config.split_names_enabled:
                        name_parts = name.split()
                        if len(name_parts) > 1:
                            for part in name_parts:
                                augmented_data.append(f"{label} {part}")

                    # Esto es de textattack, creo que será buena idea... veamos.
                    if self.config.swap_augmenter:
                        swapped_names = self.config.swap_augmenter.augment(name)
                        augmented_data.extend([f"{label} {aug_name}" for aug_name in swapped_names])

                    if self.config.insert_augmenter:
                        insertions_names = self.config.insert_augmenter.augment(name)
                        augmented_data.extend([f"{label} {aug_name}" for aug_name in insertions_names])

                    if self.config.delete_augmenter:
                        deletions_names = self.config.delete_augmenter.augment(name)
                        augmented_data.extend([f"{label} {aug_name}" for aug_name in deletions_names])

            return augmented_data

        def _internal_swap(self, name):
            name_chars = list(name)

            # solo internos, evito tocar el primer y último caracter
            for i in range(1, len(name_chars) - 2):
                if random.random() < 0.5:
                    name_chars[i], name_chars[i + 1] = name_chars[i + 1], name_chars[i]

            return ''.join(name_chars)

    def run_pipeline(self):
        if self.augmentation_config.get("enabled", False):
            data = self.augment()
        else:
            data = self.data
        # TODO: Investigar bien esto!
        # mejores resultados sin eliminar duplicados...
        # no creo que sea bueno...
        if self.unique_names:
            data = list(set(data))

        self._save_data(data)
        self._stratify_data(data)

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
                test_size = int(self.test_pct * total_items)

                train_lines.extend(items[:train_size])
                dev_lines.extend(items[train_size:train_size + dev_size])
                test_lines.extend(items[train_size + dev_size:train_size + dev_size + test_size])

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
