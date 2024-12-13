"""
*****************************************************
 * Universidad del País Vasco (UPV/EHU)
 * Facultad de Informática - Donostia-San Sebastián
 * Asignatura: Procesamiento de Lenguaje Natural
 * Proyecto: Lore Nexus
 *
 * File: lorenexus.py
 * Author: geru-scotland
 * GitHub: https://github.com/geru-scotland
 * Description:
 *****************************************************
"""
import os
from abc import ABC, abstractmethod
from datetime import datetime

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from paths import DATA_OUTPUT_DIR


class LoreNexusWrapper(ABC):
    def __init__(self, mode="train", config_info_dump="data_config.info"):
        """
        """
        self._mode = mode
        self._model = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        config_dump_path = os.path.join(DATA_OUTPUT_DIR, config_info_dump)
        if config_dump_path:
            try:
                with open(config_dump_path, 'r', encoding='utf-8') as file:
                    self.config_dump_info = file.read()
            except FileNotFoundError:
                self.config_dump_info = "Configuration file not found."
            except Exception as e:
                self.config_dump_info = f"Error reading configuration file: {e}"

    @staticmethod
    def _train_mode_only(method):
        def wrapper(self, *args, **kwargs):
            if self._mode == "train":
                return method(self, *args, **kwargs)
            else:
                print(f"Method '{method.__name__}' is only available in 'train' mode.")
        return wrapper

    @abstractmethod
    def _load_data(self, data_path):
        """
        """
        pass

    @abstractmethod
    def _create_vocab(self):
        """
        """
        pass

    @abstractmethod
    def train(self):
        """
        """
        pass

    @abstractmethod
    def evaluate(self, verbose, model=None):
        """
        """
        pass

    @abstractmethod
    def predict_name(self, name):
        """
        """
        pass

    # Esto de logger, fuera. Tiene que ir en la clase base LoreNexusWrapper
    def _plot_and_log_results(self, epoch_stats, epochs, total_train_samples, total_dev_samples, train_losses, validation_losses, hyperparams, best_results, conf_matrix = None):
        sns.set(style="whitegrid", palette="muted")
        epochs_range = list(range(1, epochs + 1))

        plt.figure(figsize=(12, 6))
        plt.plot(epochs_range, train_losses, label='Training Loss', linewidth=2.5, marker='o', markersize=7)
        plt.plot(epochs_range, validation_losses, label='Validation Loss', linewidth=2.5, marker='s', markersize=7)
        plt.fill_between(epochs_range, train_losses, validation_losses, color="lightcoral", alpha=0.2)

        plt.title("Training and Validation Loss Across Epochs", fontsize=18)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.xticks(epochs_range)
        plt.legend(loc="upper right", fontsize=12, bbox_to_anchor=(1.15, 1))
        plt.grid(visible=True, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

        hyperparams_text = '\n'.join([f"{k}: {v}" for k, v in hyperparams.items()])
        plt.text(0.95, 0.05, hyperparams_text, fontsize=10, ha='right', va='bottom', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))

        plt.tight_layout()

        timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")

        log_dir = f"logs/{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        filename = os.path.join(log_dir, f"train_run_{timestamp}.png")
        plt.savefig(filename)

        print(f"Metrics file stored as '{filename}'")

        cm_filename = None
        if conf_matrix is not None:
            plt.figure(figsize=(10, 7))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title('Best Confusion Matrix')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.tight_layout()

            cm_filename = os.path.join(log_dir, f"conf_matrix_{timestamp}.png")
            plt.savefig(cm_filename)
            print(f"Confusion matrix figure saved as {cm_filename}")

        info_file_path = filename.replace(".png", ".log")

        with open(info_file_path, "w", encoding="utf-8") as f:
            f.write("*****************************************************\n")
            f.write("* Dataset Sizes\n")
            f.write("*****************************************************\n")
            f.write(f"Training samples: {total_train_samples}\n")
            f.write(f"Validation samples: {total_dev_samples}\n")

            f.write("*****************************************************\n")
            f.write("* Best Results\n")
            f.write("*****************************************************\n")
            f.write(f"Best validation accuracy: {best_results['accuracy']:.4f}\n")
            f.write(f"Best Epoch: {best_results['epoch']}\n")
            f.write(f"Best training loss: {best_results['train_loss']:.4f}\n")
            f.write(f"Best validation loss: {best_results['val_loss']:.4f}\n\n")

            f.write("*****************************************************\n")
            f.write("* Hyperparameters\n")
            f.write("*****************************************************\n")

            for key, value in hyperparams.items():
                f.write(f"{key}: {value}\n")

            f.write("\n*****************************************************\n")
            f.write("* Best Classification Report\n")
            f.write("*****************************************************\n")
            f.write(best_results.get('report', 'No report available'))

            f.write("\n*****************************************************\n")
            f.write("* SK Learn Accuracy score\n")
            f.write("*****************************************************\n")
            f.write(f"Best score: {best_results['acc_score_sklearn']:.4f}\n\n")

            f.write("\n*****************************************************\n")
            f.write("* Data Configuration\n")
            f.write("*****************************************************\n")
            f.write(self.config_dump_info)
            f.write("\n")

            f.write("\n*****************************************************\n")
            f.write("* Epoch Output\n")
            f.write("*****************************************************\n")
            for epoch_stat in epoch_stats:
                f.write(f"Epoch {epoch_stat['epoch']}:\n")
                f.write(f"  Training Loss: {epoch_stat['train_loss']:.4f}\n")
                f.write(f"  Validation Loss: {epoch_stat['val_loss']:.4f}\n")
                f.write(f"  Validation Accuracy: {epoch_stat['val_accuracy']:.4f}\n")
                f.write(f"  Learning Rate: {epoch_stat['learning_rate']:.6f}\n")
                f.write("\n")

            if conf_matrix is not None:
                f.write("*****************************************************\n")
                f.write("* Confusion Matrix\n")
                f.write("*****************************************************\n")
                cm_str = np.array2string(conf_matrix, separator=', ')
                f.write(f"{cm_str}\n")

        print(f"Configuration and results file saved as: {info_file_path}")

