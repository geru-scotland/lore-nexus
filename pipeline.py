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

config = Config('config.json')

wikidata_config = config.get_dataset("Wikidata")

if wikidata_config:
    wikidata_processor = WikidataProcessor(
        input_file=f"{wikidata_config['path']}/{wikidata_config['input_folder']}/{wikidata_config['dataset_file']}",
        output_folder=f"{wikidata_config['path']}/{wikidata_config['output_folder']}",
        labels_file=f"{wikidata_config['path']}/labels/labels.txt"
    )
    wikidata_processor.process_data(DatasetFormats.FAST_TEXT)


mythdata_config = config.get_dataset("Mythdata")

if mythdata_config:
    mythdata_processor = MythdataProcessor(
        input_file=f"{mythdata_config['path']}/{mythdata_config['input_folder']}/{mythdata_config['dataset_file']}",
        output_file=f"{mythdata_config['path']}/{mythdata_config['output_folder']}/{mythdata_config['output_file']}"
    )
    mythdata_processor.process_data()


data_processor_config = config.get_data_processor_config()
base_path = data_processor_config["path"]

data_processor = DataProcessor(
    datasets=config.config["datasets"],
    labels=data_processor_config["labels"],
    output_folder=f"{base_path}/{data_processor_config['output_folder']}",
    output_file=data_processor_config["output_file"],
    train_file=data_processor_config["train_file"],
    dev_file=data_processor_config["dev_file"],
    test_file=data_processor_config["test_file"]
)
data_processor.run_pipeline()