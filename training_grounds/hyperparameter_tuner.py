"""
*****************************************************
 * Universidad del País Vasco (UPV/EHU)
 * Facultad de Informática - Donostia-San Sebastián
 * Asignatura: Procesamiento de Lenguaje Natural
 * Proyecto: Lore Nexus
 *
 * File: hyperparameter_tuner.py
 * Author: geru-scotland
 * GitHub: https://github.com/geru-scotland
 * Description:
 *****************************************************
"""
import argparse
import json

import logging
import os
import shutil
from pathlib import Path
from datetime import datetime

from models.flair.model import LoreNexusFlairModel
from models.pytorch.model import LoreNexusPytorchModel
from paths import DATA_OUTPUT_DIR


class HyperparameterTuner:
    """
    TODO: Que busque iterativamente el mejor valor de algunos hiperparámetros, como el learning rate y weight decay.
    """

    class Experiment:
        def __init__(self, model, model_name, params, logger):
            self.model = model
            self.model_name = model_name
            self.params = params
            self.logger = logger
            self.result = None

        def run(self, output_path):
            self.logger.info(f"Starting training with params: {self.params}")
            self.model.train(output_path=output_path, save_model=False, log_results=True, **self.params)
            self.result = self.model.evaluate()
            return self.result

    def __init__(self, models, param_grid, log_dir='logs', config_info_dump="data_config.info"):
        self.models = models
        self.param_grid = param_grid
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.best_results = {}

        config_dump_path = os.path.join(DATA_OUTPUT_DIR, config_info_dump)
        if config_dump_path:
            try:
                with open(config_dump_path, 'r', encoding='utf-8') as file:
                    self.config_info = file.read()
            except FileNotFoundError:
                self.config_info = "Configuration file not found."
            except Exception as e:
                self.config_info = f"Error reading configuration file: {e}"

    def run(self):
        for model_name, model in self.models.items():
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

            logger = self._get_model_logger(model_name, timestamp)

            best_accuracy = 0.0
            best_params = None
            best_report = None

            for params in self.param_grid:
                output_path = Path(f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                experiment = self.Experiment(model, model_name, params, logger)
                accuracy, report = experiment.run(output_path=output_path)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = params
                    best_report = report

                self.log_results(model_name, params, accuracy, report, logger)

            self.best_results[model_name] = {
                "accuracy": best_accuracy,
                "params": best_params,
                "report": best_report
            }

            best_result_message = (
                    f"\nBest Result for {model_name}:\n"
                    f"Best Accuracy: {best_accuracy:.4f}\n"
                    f"Best Hyperparameters: {best_params}\n"
                    f"Best Classification Report:\n{best_report}\n"
                    + "=" * 50
            )

            print(best_result_message)
            logger.info(best_result_message)

        final_summary = (
                f"\nConfiguration Used:\n"
                f"{self.config_info}\n"
                + "=" * 50 +
                "\nFinal Summary of Best Results:\n" + "=" * 50
        )

        for model_name, result in self.best_results.items():
            final_summary += (
                    f"\n\nModel: {model_name}\n"
                    f"Best Accuracy: {result['accuracy']:.4f}\n"
                    f"Best Hyperparameters: {result['params']}\n"
                    f"Best Classification Report:\n{result['report']}\n"
                    + "=" * 50
            )

        # Aparcao, muestro en pantalla y lanzo best results finales al logger, por modelo.
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(final_summary)
        for model_name in self.best_results:
            self._get_model_logger(model_name, timestamp).info(final_summary)

    # en un principio parecia buena idea tener un logger por modelo, pero igual lo quito
    # TODO: Echar una pensada a esto
    def _get_model_logger(self, model_name, timestamp):
        timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
        arena = f"{self.log_dir}/arena_{timestamp}"
        os.makedirs(f"{arena}", exist_ok=True)
        log_file = Path(arena) / f"{model_name}_hyperparameter_tuning_{timestamp}.log"
        logger = logging.getLogger(model_name)

        if not logger.hasHandlers():
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(f'%(asctime)s - {model_name} - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def log_results(self, model_name, params, accuracy, report, logger):

        logger.info(f"Model: {model_name}")
        logger.info(f"Hyperparameters: {params}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("Classification Report:")
        logger.info(report)
        logger.info("=" * 50)


pytorch_model = LoreNexusPytorchModel(mode="train")
flair_model = LoreNexusFlairModel(mode="train")

models = {
    "PytorchModel": pytorch_model,
    "FlairModel": flair_model
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Load models and execute tuner")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["pytorch", "flair"],
        help="Select the model to load: 'pytorch' or 'flair'",
    )
    args = parser.parse_args()

    if args.model == "pytorch":
        selected_model = {"PytorchModel": LoreNexusPytorchModel(mode="train")}
    elif args.model == "flair":
        selected_model = {"FlairModel": LoreNexusFlairModel(mode="train")}
    else:
        selected_model = {
            "PytorchModel": LoreNexusPytorchModel(mode="train"),
            "FlairModel": LoreNexusFlairModel(mode="train"),
        }

    with open("param_grids.json", "r") as f:
        param_grid = json.load(f)

    # TODO: Hacer que busque el mejor valor de learning rate (generar una grid con rango de valores en lugar de cargarla)
    tuner = HyperparameterTuner(selected_model, param_grid)
    tuner.run()

    # Hack feo para borrar los modelos de flair que se generan en cada iteración
    # no consigo encontrar la manera de que flair no lo guarde en el train
    current_dir = os.getcwd()
    for item in os.listdir(current_dir):
        full_path = os.path.join(current_dir, item)
        if os.path.isdir(full_path) and item.startswith("FlairModel"):
            shutil.rmtree(full_path)
            print(f"Deleting: {full_path}")