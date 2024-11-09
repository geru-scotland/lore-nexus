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

from dataset.preprocessing.apis.mythology.mythdata import MythdataProcessor
from dataset.preprocessing.apis.wikidata.wikidata import WikidataProcessor, DatasetFormats
from dataset.preprocessing.data_processor import DataProcessor


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

class DataPipeline:
    def __init__(self, config_path='config.json'):
        self.config = Config(config_path)

    def process_wikidata(self):
        wikidata_config = self.config.get_dataset("Wikidata")
        if wikidata_config:
            try:
                wikidata_processor = WikidataProcessor(
                    input_file=f"{wikidata_config['path']}/{wikidata_config['input_folder']}/{wikidata_config['dataset_file']}",
                    output_folder=f"{wikidata_config['path']}/{wikidata_config['output_folder']}",
                    labels_file=f"{wikidata_config['path']}/labels/labels.txt"
                )
                wikidata_processor.process_data(DatasetFormats.FAST_TEXT)
                print("Wikidata processing completed.")
            except Exception as e:
                print(f"Error processing Wikidata: {e}")

    def process_mythdata(self):
        mythdata_config = self.config.get_dataset("Mythdata")
        if mythdata_config:
            try:
                mythdata_processor = MythdataProcessor(
                    input_file=f"{mythdata_config['path']}/{mythdata_config['input_folder']}/{mythdata_config['dataset_file']}",
                    output_file=f"{mythdata_config['path']}/{mythdata_config['output_folder']}/{mythdata_config['output_file']}"
                )
                mythdata_processor.process_data()
                print("Mythdata processing completed.")
            except Exception as e:
                print(f"Error processing Mythdata: {e}")

    def run_data_processor(self):
        data_processor_config = self.config.get_data_processor_config()
        base_path = data_processor_config["path"]

        try:
            data_processor = DataProcessor(
                datasets=self.config.config["datasets"],
                labels=data_processor_config["labels"],
                output_folder=f"{base_path}/{data_processor_config['output_folder']}",
                output_file=data_processor_config["output_file"],
                train_file=data_processor_config["train_file"],
                dev_file=data_processor_config["dev_file"],
                test_file=data_processor_config["test_file"]
            )
            data_processor.run_pipeline()
            print("Data processing pipeline completed.")
        except Exception as e:
            print(f"Error in data processing pipeline: {e}")

    def run(self):
        # TODO: Mejorar un poco el output, formato, colores y sincronizar con cada proceso
        # que ahora lo he dejado con spam de prints
        print("Starting Data Pipeline...")
        self.process_wikidata()
        self.process_mythdata()
        self.run_data_processor()
        print("Data Pipeline completed.")


if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run()