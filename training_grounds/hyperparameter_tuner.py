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
import json

import logging
from pathlib import Path
from datetime import datetime

from models.flair.model import LoreNexusFlairModel
from models.pytorch.model import LoreNexusPytorchModel

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
            self.model.train(output_path=output_path, save_model=False, **self.params)
            self.result = self.model.evaluate()
            return self.result

    def __init__(self, models, param_grid, log_dir='logs'):
        self.models = models
        self.param_grid = param_grid
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.best_results = {}

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

        final_summary = "\nFinal Summary of Best Results:\n" + "=" * 50
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

    def _get_model_logger(self, model_name, timestamp):
        log_file = self.log_dir / f"{model_name}_hyperparameter_tuning_{timestamp}.log"
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
    with open("param_grids.json", "r") as f:
        param_grid = json.load(f)

    # TODO: Hacer que busque el mejor valor de learning rate (generar una grid con rango de valores en lugar de cargarla)
    tuner = HyperparameterTuner(models, param_grid)
    tuner.run()
