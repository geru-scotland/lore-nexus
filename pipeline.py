"""
*****************************************************
 * Universidad del País Vasco (UPV/EHU)
 * Facultad de Informática - Donostia-San Sebastián
 * Asignatura: Procesamiento de Lenguaje Natural
 * Proyecto: Lore Nexus
 *
 * File: pipeline.py
 * Author: geru-scotland
 * GitHub: https://github.com/geru-scotland
 * Description:
 *****************************************************
"""

import json
import os

from dataset.preprocessing.apis.mythology.mythdata import MythdataProcessor
from dataset.preprocessing.apis.slurs.slurs import SlursProcessor
from dataset.preprocessing.apis.wikidata.wikidata import WikidataProcessor, DatasetFormats
from dataset.preprocessing.data_processor import DataProcessor
from dataset.preprocessing.ner.ner_corpus_builder import EntityCorpusBuilder
from paths import APIS_DIR, NER_DIR, DATA_OUTPUT_DIR


class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = json.load(file)

    def get_dataset(self, name):
        """
        """
        for dataset in self.config.get("datasets", []):
            if dataset["name"] == name:
                return dataset
        return None

    def get_data_processor_config(self):
        """
        """
        return self.config.get("data_processor", {})

    def dump_config_to_file(self):
        """
        Generates a .info file with the configuration used for processing.
        """
        data_processor_config = self.get_data_processor_config()
        config_dump_path = os.path.join(DATA_OUTPUT_DIR, data_processor_config["config_dump"])

        try:
            with open(config_dump_path, 'w', encoding='utf-8') as file:
                file.write("*****************************************************\n")
                file.write("* Data Processing Configuration\n")
                file.write("*****************************************************\n")
                for key, value in data_processor_config.items():
                    if key == "augmentation":
                        file.write(f"{key}:\n")
                        for aug_key, aug_value in value.items():
                            file.write(f"  {aug_key}: {aug_value}\n")
                    else:
                        file.write(f"{key}: {value}\n")
                file.write("\nDatasets:\n")
                for dataset in self.config.get("datasets", []):
                    file.write(f"  - {dataset['name']}:\n")
                    for k, v in dataset.items():
                        file.write(f"      {k}: {v}\n")
            print(f"Configuration dumped successfully to {config_dump_path}")
        except Exception as e:
            print(f"Error dumping configuration to file: {e}")

class DataPipeline:
    def __init__(self, config_path='config.json'):
        self.config = Config(config_path)

    def process_wikidata(self):
        wikidata_config = self.config.get_dataset("Wikidata")
        if wikidata_config:
            try:
                wikidata_path = os.path.join(APIS_DIR, wikidata_config['path'])
                input_file_path =  os.path.join(str(wikidata_path), wikidata_config['input_folder'], wikidata_config['dataset_file'])
                output_file_path = os.path.join(str(wikidata_path), wikidata_config['output_folder'], wikidata_config['output_file'])
                labels_path = os.path.join(str(wikidata_path), wikidata_config['labels_folder'], wikidata_config['labels_file'])

                wikidata_processor = WikidataProcessor(
                    input_file=f"{input_file_path}",
                    output_file=f"{output_file_path}",
                    labels_file=f"{labels_path}"
                )

                wikidata_processor.process_data(DatasetFormats.FAST_TEXT)
                print("Wikidata processing completed.")
            except Exception as e:
                print(f"Error processing Wikidata: {e}")

    def process_mythdata(self):
        mythdata_config = self.config.get_dataset("Mythdata")
        if mythdata_config:
            try:

                mythdata_path = os.path.join(APIS_DIR, mythdata_config['path'])
                input_file_path =  os.path.join(str(mythdata_path), mythdata_config['input_folder'], mythdata_config['dataset_file'])
                output_file_path = os.path.join(str(mythdata_path), mythdata_config['output_folder'], mythdata_config['output_file'])

                mythdata_processor = MythdataProcessor(input_file=f"{input_file_path}", output_file=f"{output_file_path}")

                mythdata_processor.process_data()
                print("Mythdata processing completed.")
            except Exception as e:
                print(f"Error processing Mythdata: {e}")

    def process_nerdata(self):
        nerdata_config = self.config.get_dataset("NERdata")
        if nerdata_config:
            try:
                mythdata_processor = EntityCorpusBuilder()

                ner_input_path = os.path.join(NER_DIR, nerdata_config['input_folder'])
                ner_output_file = os.path.join(NER_DIR, nerdata_config['output_folder'], nerdata_config["output_file"])

                mythdata_processor.label_and_filter_entities(input_dir=f"{ner_input_path}", output_file=f"{ner_output_file}")

                print("NERdata processing completed.")
            except Exception as e:
                print(f"Error processing NERdata: {e}")

    def process_slurs(self):
        slurs_config = self.config.get_dataset("Slurs")
        if slurs_config:
            try:
                slurs_processor = SlursProcessor(
                    input_file=os.path.join(APIS_DIR, slurs_config['path'], slurs_config['input_folder'], slurs_config['dataset_file']),
                    output_file=os.path.join(APIS_DIR, slurs_config['path'], slurs_config['output_folder'], slurs_config['output_file'])
                )
                slurs_processor.process()
                print("Slurs processing completed.")
            except Exception as e:
                print(f"Error processing Slurs: {e}")

    def run_data_processor(self):
        data_processor_config = self.config.get_data_processor_config()

        try:
            data_processor = DataProcessor(
                datasets=self.config.config["datasets"],
                labels=data_processor_config["labels"],
                augmentation=data_processor_config["augmentation"],
                output_file=data_processor_config["output_file"],
                train_file=data_processor_config["train_file"],
                dev_file=data_processor_config["dev_file"],
                test_file=data_processor_config["test_file"],
                train_pct=data_processor_config["train_size"],
                dev_pct=data_processor_config["dev_size"],
                test_pct=data_processor_config["test_size"],
            )
            data_processor.run_pipeline()
            self.config.dump_config_to_file()
            print("Data processing pipeline completed.")
        except Exception as e:
            print(f"Error in data processing pipeline: {e}")

    def run(self):
        # TODO: Mejorar un poco el output, formato, colores y sincronizar con cada proceso
        # que ahora lo he dejado con spam de prints
        print("Starting Data Pipeline...")
        self.process_wikidata()
        self.process_mythdata()
        self.process_nerdata()
        self.process_slurs()
        self.run_data_processor()
        print("Data Pipeline completed.")


if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run()