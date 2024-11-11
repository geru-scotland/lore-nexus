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
from abc import ABC, abstractmethod

import torch

class LoreNexusWrapper(ABC):
    def __init__(self, mode="train"):
        """
        """
        self._mode = mode
        self._model = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    def train(self, train_data):
        """
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        """
        pass

    @abstractmethod
    def predict_name(self, name):
        """
        """
        pass

