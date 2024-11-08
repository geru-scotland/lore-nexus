"""
*****************************************************
 * Universidad del País Vasco (UPV/EHU)
 * Facultad de Informática - Donostia-San Sebastián
 * Asignatura: Procesamiento de Lenguaje Natural
 * Proyecto: Lore Nexus
 *
 * File: train.py
 * Author: geru-scotland
 * GitHub: https://github.com/geru-scotland
 * Description:
 *****************************************************
"""

from flair.datasets import ClassificationCorpus
from flair.data import Corpus
from flair.embeddings import CharacterEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from torch.optim import AdamW
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_folder = 'data/'

corpus: Corpus = ClassificationCorpus(data_folder,
                                      train_file='train.txt',
                                      dev_file='dev.txt',
                                      test_file='test.txt',
                                      label_type='class')

label_dict = corpus.make_label_dictionary(label_type='class')

embedding = DocumentRNNEmbeddings(
    embeddings=[
        # Igual probar con los embeddings de Flair preentrenados, pero quizá rompe la coherencia de caracteres
        # cacharrear a ver
        CharacterEmbeddings()
    ],
    hidden_size=768,
    rnn_type='LSTM',
    dropout=0.2
)

embedding.to(device)

classifier = TextClassifier(embeddings=embedding,
                            label_dictionary=label_dict,
                            label_type='class',
                            multi_label=False)

classifier.to(device)

trainer = ModelTrainer(classifier, corpus)

# No va mal del todo este setup, creo que es cuestión de datos.
trainer.train('resources/taggers/universe_classifier',
              learning_rate=0.001,
              mini_batch_size=16,
              max_epochs=30,
              optimizer=AdamW,
              weight_decay=0.01,
              embeddings_storage_mode='gpu' if torch.cuda.is_available() else 'cpu',
)
