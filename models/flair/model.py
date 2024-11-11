"""
*****************************************************
 * Universidad del País Vasco (UPV/EHU)
 * Facultad de Informática - Donostia-San Sebastián
 * Asignatura: Procesamiento de Lenguaje Natural
 * Proyecto: Lore Nexus
 *
 * File: model.py
 * Author: geru-scotland
 * GitHub: https://github.com/geru-scotland
 * Description:
 *****************************************************
"""
from abc import ABC
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from models.lorenexus.lorenexus import LoreNexusWrapper

from flair.datasets import ClassificationCorpus
from flair.data import Sentence
from flair.embeddings import CharacterEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from torch.optim import AdamW
import torch


class LoreNexusFlairModel(LoreNexusWrapper, ABC):
    def __init__(self, mode="train", data_folder='data/', model_path=None):
        """
        """
        super().__init__(mode)

        self._mode = mode

        if self._mode == "train":
            self._corpus = self._load_data(data_folder)
            self._label_dict = self._create_vocab()
            self._embedding = self._initialize_embeddings()
            self._classifier = self._initialize_classifier()
        elif self._mode == "evaluate":
            self._corpus = self._load_data(data_folder)
            try:
                self._classifier = TextClassifier.load(model_path)
                print("Model loaded for evaluation: ")
                print(f"Model: {model_path}")
            except Exception as e:
                print(f"Error while loading the model: {e}")
                sys.exit(1)
        elif self._mode == "cli_app":
            # Solo una vez, que al cabrón a veces le cuesta levantarse
            try:
                self._classifier = TextClassifier.load(model_path)
                print("LoreNexus (BiLSTM-Flair) loaded for CLI predictions: ")
                print(f"Model: {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                sys.exit(1)

    def _load_data(self, data_folder):
        """
        """
        return ClassificationCorpus(
            data_folder,
            train_file='train.txt',
            dev_file='dev.txt',
            test_file='test.txt',
            label_type='class'
        )

    @LoreNexusWrapper._train_mode_only
    def _create_vocab(self):
        """
        """
        return self._corpus.make_label_dictionary(label_type='class')

    @LoreNexusWrapper._train_mode_only
    def _initialize_embeddings(self):
        """
        """
        embedding = DocumentRNNEmbeddings(
            embeddings=[CharacterEmbeddings()],
            hidden_size=768,
            rnn_type='LSTM',
            bidirectional=True,
            dropout=0.2
        )
        embedding.to(self._device)
        return embedding

    @LoreNexusWrapper._train_mode_only
    def _initialize_classifier(self):
        """
        """
        classifier = TextClassifier(
            embeddings=self._embedding,
            label_dictionary=self._label_dict,
            label_type='class',
            multi_label=False
        )
        classifier.to(self._device)
        return classifier

    @LoreNexusWrapper._train_mode_only
    def train(self, output_path='resources/taggers/universe_classifier', save_model=True, lr=0.001, batch_size=32,
              epochs=10, weight_decay=0.01, hidden_dim=256, embeddings_dim=100, num_layers=1, dropout=0.2):
        """
        TODO: Cargar del config.json
        """
        trainer = self._setup_trainer()
        trainer.train(
            output_path,
            learning_rate=lr,
            mini_batch_size=batch_size,
            max_epochs=epochs,
            optimizer=AdamW,
            weight_decay=weight_decay,
            embeddings_storage_mode='gpu' if torch.cuda.is_available() else 'cpu',
            save_final_model=save_model
        )

    @LoreNexusWrapper._train_mode_only
    def _setup_trainer(self):
        """
        """
        return ModelTrainer(self._classifier, self._corpus)

    def evaluate(self, output_path='resources/taggers/universe_classifier', batch_size=32, verbose=True):
        """
        """
        if self._classifier is None:
            print("The model is not initialized.")
            return

        result = self._classifier.evaluate(
            self._corpus.test,
            mini_batch_size=batch_size,
            embeddings_storage_mode='gpu' if torch.cuda.is_available() else 'cpu',
            gold_label_type='class'
        )

        accuracy = result.main_score

        if verbose:
            print(result.detailed_results)
            print(f"Test Accuracy: {accuracy:.4f}")

        return accuracy, result.detailed_results

    def predict_name(self, name):
        """
        """
        sentence = Sentence(name.lower())
        self._classifier.predict(sentence, return_probabilities_for_all_classes=True)

        top_labels = sorted(sentence.labels, key=lambda label: label.score, reverse=True)[:4]

        results = {}
        for i, label in enumerate(top_labels):
            results[f"Prediction {i + 1}"] = (label.value, label.score)

        return results

# model = LoreNexusFlairModel(mode="train",
#                             model_path='resources/taggers/universe_classifier/best-model.pt')
# model.train()
# model.evaluate()