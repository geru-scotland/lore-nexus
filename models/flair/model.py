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

from models.lorenexus.lorenexus import LoreNexusWrapper

from flair.datasets import ClassificationCorpus
from flair.data import Corpus, Sentence
from flair.embeddings import CharacterEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from torch.optim import AdamW
import torch


class LoreNexusFlairModel(LoreNexusWrapper, ABC):
    def __init__(self, mode="train", data_folder='data/'):
        """
        """
        super().__init__(mode)

        self._mode = mode

        if self._mode == "train":
            self._corpus = self._load_data(data_folder)
            self._label_dict = self._create_vocab()
            self._embedding = self._initialize_embeddings()
            self._classifier = self._initialize_classifier()

    @LoreNexusWrapper._train_mode_only
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
    def train(self, output_path='resources/taggers/universe_classifier', lr=0.001, batch_size=32, epochs=10):
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
            weight_decay=0.01,
            embeddings_storage_mode='gpu' if torch.cuda.is_available() else 'cpu'
        )

    @LoreNexusWrapper._train_mode_only
    def _setup_trainer(self):
        """
        """
        return ModelTrainer(self._classifier, self._corpus)

    def evaluate(self, eval_data):
        """
        """
        pass

    def predict(self, model_path='resources/taggers/universe_classifier/best-model.pt', name=""):
        """
        """
        if self._mode == "eval":
            classifier = TextClassifier.load(model_path)
            sentence = Sentence(name.lower())

            classifier.predict(sentence, return_probabilities_for_all_classes=True)

            top_labels = sorted(sentence.labels, key=lambda label: label.score, reverse=True)[:4]

            for i, label in enumerate(top_labels, start=1):
                print(f"Prediction {i}: {label.value} with confidence {label.score:.4f}")


ln_flair_model = LoreNexusFlairModel(mode="eval")
ln_flair_model.predict(name="driz dourdenn")
