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
import os.path
from abc import ABC
import sys
from collections import Counter
from datetime import datetime
from enum import Enum
from pathlib import Path

import torch.utils.data
import torch.nn.functional as F
from torch.nn import init

from paths import DATA_OUTPUT_DIR

sys.path.append(str(Path(__file__).resolve().parents[2]))

from models.lorenexus.lorenexus import LoreNexusWrapper

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch import tensor

class SpecialTokens(Enum):
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __str__(self):
        return self.value

class LoreNexusPytorchModel(LoreNexusWrapper, ABC):

    class CharacterVocab:
        def __init__(self):
            self.char2index = {} # recuerda, este fundamental para entrada
            self.index2char = {} # y este para salida, argmax de softmax y vienes a este

        def __len__(self):
            return len(self.char2index)

        def __getitem__(self, key):
            if isinstance(key, int):
                return self.index2char[key]
            elif isinstance(key, str):
                return self.char2index[key]
            else:
                raise ValueError("Key must be either int or str")

        def _add_special_tokens(self):
            self.char2index[str(SpecialTokens.PAD)] = 0
            self.index2char[0] = str(SpecialTokens.PAD)
            self.char2index[str(SpecialTokens.UNK)] = 1
            self.index2char[1] = str(SpecialTokens.UNK)

        def encode(self, word):
            tokenized_chars = []
            for char in word:
                if char not in self.char2index:
                    index = len(self)
                    self.char2index[char] = index
                    self.index2char[index] = char
                tokenized_chars.append(self.char2index[char])
            return tokenized_chars

        def get_index(self, char=None):
            return self.char2index.get(char, self.char2index[str(SpecialTokens.UNK)])

        def get_char(self, index=None):
            return self.index2char.get(index, str(SpecialTokens.UNK))

        def decode(self, indices):
            if isinstance(indices, int):
                indices = [indices]
            return "".join([self.index2char[index] for index in indices if index in self.index2char])

        def build_vocab(self, data):
            self._add_special_tokens()
            for word in data:
                self.encode(word)

    class LabelEndoder:
        def __init__(self):
            self.label_to_index = {}
            self.index_to_label = {}

        def __len__(self):
            return len(self.label_to_index)

        def __getitem__(self, item):
            if isinstance(item, str):
                return self.label_to_index.get(item, None)
            elif isinstance(item, int):
                return self.index_to_label.get(item, None)
            raise ValueError("Item must be either str or int")

        def encode(self, label):
            if label not in self.label_to_index:
                index = len(self)
                self.label_to_index[label] = index
                self.index_to_label[index] = label
            return self.label_to_index.get(label, None)

        def decode(self, index):
            return self.index_to_label.get(index, None)

        def unpack_indexes(self):
            return self.index_to_label.keys(), self.index_to_label.values()

        def unpack_labels(self):
            return self.label_to_index.keys(), self.label_to_index.values()



    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data, char_vocab, max_length=40, build_vocab=False):
            self._max_length = max_length
            self.names, self.labels = data # desempaqueto, estaba gestionando esto mal
            assert len(self.names) == len(self.labels), "names and labels need to have same length"
            self._char_vocab = char_vocab
            if build_vocab:
                self.tokenize_data()


        def __len__(self):
            return len(self.names)

        def __getitem__(self, index):
            """
            Tengo que devolver el nombre tokenizado y label, esto lo llamará el DataLoader
            para pasar inputs y targets al modelo.
            Tokenizo el nombre y lo corto si es más largo que el máximo
            TODO: Mirar como determinar el máximo, ahora pongo 40 a lo loco
            Quizá simplemente coger el máximo de entre todas las secuencias
            """

            name, label = self.names[index], self.labels[index]

            tokenized_name = self._char_vocab.encode(name)
            padded_vector = tokenized_name[:self._max_length] # corto, si faltan va a haber que rellenar
            padding_size = self._max_length - len(padded_vector)

            if padding_size > 0:
                padded_vector.extend([self._char_vocab.get_index("<PAD>")]*padding_size)

            # devuelvo 2 tensores
            return tensor(padded_vector, dtype=torch.long), tensor(label, dtype=torch.long)

        def tokenize_data(self):
            self._char_vocab.build_vocab(self.names)

    class BiLSTMCharacterLevel(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
            super().__init__()

            # En este caso, no es necesario utilizar nn.Parameters, no utilizo
            # tensores custom, estos los gestiona internamente PyTorch
            self.character_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

            self.bi_lstm = torch.nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional=True,
                batch_first=True, # por compatibilidad, con shape (batch_size, sequence_len, embedding_dim)
                dropout=dropout
            )

            # OJO! bidireccional, por eso el *2 en el hidden_dim, uno para cada dirección
            self.fc = torch.nn.Linear(hidden_dim*2, output_dim)
            self.apply(self._init_weights)

        def forward(self, input_tensor):

            embedded_tensor = self.character_embeddings(input_tensor)
            lstm_out, (hidden_state, cell) = self.bi_lstm(embedded_tensor)

            # Básicamente estoy concatenando los estados ocultos finales de las dos direcciones
            hidden = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)
            out = self.fc(hidden)

            return out

        def _init_weights(self, module):
            if isinstance(module, torch.nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, torch.nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        init.kaiming_uniform_(param)
                    elif 'bias' in name:
                        init.zeros_(param)

    def __init__(self, mode="train", data_folder='data/', model_path="checkpoints/best_model.pth", config_info_dump="data_config.info"):
        """
        """
        super().__init__(mode, config_info_dump)

        self._mode = mode
        self._char_vocab = None
        self._label_encoder = None
        self._data_folder = data_folder
        self._hyperparams = {}

        if self._mode == "cli_app":
            # Solo una vez, que al cabrón a veces le cuesta levantarse
            try:
                self.load_model(model_path)
                print("LoreNexus (BiLSTM-Pytorch) loaded for CLI predictions: ")
                print(f"Model: {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                sys.exit(1)

    @LoreNexusWrapper._train_mode_only
    def _create_vocab(self):
        return self.CharacterVocab()

    @LoreNexusWrapper._train_mode_only
    def _load_data(self, data_folder, splits=None):
        """
        """
        data = {}
        if splits is None:
            splits = ['train', 'dev', 'test']

        for split in splits:
            file_path = os.path.join(DATA_OUTPUT_DIR, f"{split}.txt")
            names, labels = [], []

            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()

                    parts = line.split(' ', 1)
                    label = parts[0].replace('__label__', '')
                    name = parts[1]

                    # Necesito que las labels sean enteros, asigno uno a cada una
                    labels.append(self._label_encoder.encode(label))
                    names.append(name)

            data[split] = (names, labels)

        return {split: data[split] for split in splits}

    @LoreNexusWrapper._train_mode_only
    def train(self, output_path='checkpoints', save_model=True, log_results=True, lr=0.001, batch_size=32, epochs=15, weight_decay=0.02,
              hidden_dim=768, embeddings_dim=256, num_layers=1, dropout=0.2):
        """
        TODO: Cargar del config.json
        """
        # 1) Creo el vocabulario
        # 2) Cargo los datos
        # 3) Creo el modelo
        # 4) Entreno
        self._char_vocab = self._create_vocab()
        self._label_encoder = self.LabelEndoder()
        unique_labels = set()
        train_losses = []
        validation_losses = []
        epoch_stats = []

        hyperparams = {
            'lr': lr,
            'batch_size': batch_size,
            'epochs': epochs,
            'weight_decay': weight_decay,
            'hidden_dim': hidden_dim,
            'embedding_dim': embeddings_dim,
            'num_layers': num_layers,
            'dropout': dropout
        }

        self._hyperparams = hyperparams

        # Son tuplas (names, labels), cada una. Ignoro test, lo cargo al evaluar
        data_splits = self._load_data(self._data_folder, splits=['train', 'dev'])
        train_data, dev_data = data_splits['train'], data_splits['dev']

        _, labels = train_data
        for label in labels:
            unique_labels.add(label)

        train_dataset = self.Dataset(train_data, self._char_vocab, build_vocab=True)
        dev_dataset = self.Dataset(dev_data, self._char_vocab)

        label_counts = Counter(labels)
        label_list = sorted(label_counts.keys())
        total_labels = sum(label_counts.values())
        weights = []

        for label in label_list:
            weight = total_labels / label_counts[label]
            weights.append(weight)

        # imprimo labels del encoder
        indexes, label_str = self._label_encoder.unpack_indexes()
        print("Label indexes:", indexes)
        print("Label strings:", label_str)
        # y el counter ahora
        print("Label counts:", label_counts)
        class_weights = torch.tensor(weights, dtype=torch.float).to(self._device)
        print("Class weights:", class_weights)
        # Me devuelve iterables, que son los batches
        train_batches = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_batches = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

        total_train_samples = len(train_dataset)
        total_dev_samples = len(dev_dataset)

        # Creo el modelo
        self._model = self.BiLSTMCharacterLevel(
            vocab_size=len(self._char_vocab),
            embedding_dim=embeddings_dim,
            hidden_dim=hidden_dim,
            output_dim=len(unique_labels),
            n_layers=num_layers,
            dropout=dropout # si pones dropout > 0, hay que poner más capas, no solo 1 (docu de Pytorch)
        ).to(self._device)

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

        # TODO: Como tengo clases un poco desbalanceadas, utilizaré métrica F1 para que
        # mida el rendimiento un poco más completo

        best_validation_accuracy = 0.0
        best_results = {}

        for epoch in range(epochs):
            self._model.train()
            train_loss = 0.0

            for train_batch_names, train_batch_labels in train_batches:

                train_batch_names = train_batch_names.to(self._device)
                train_batch_labels = train_batch_labels.to(self._device)

                # forward
                optimizer.zero_grad() # se me olvida siempre, siempre hay que resetear gradientes después de un forward
                train_predictions = self._model(train_batch_names) # forward

                # para debuggin, sublista de predicciones y label reales, para logs
                # predicted_labels = torch.argmax(train_predictions, dim=1)
                # self.display_batch_info(train_batch_names, train_batch_labels, predicted_labels, char_vocab)

                loss = criterion(train_predictions, train_batch_labels)
                train_loss += loss.item()

                # backward
                loss.backward() # solo calcula los gradientes
                optimizer.step() # y esto ya, updatea pesos

            average_train_loss = train_loss / len(train_batches)

            # TODO: mejorar el output Y GUARDAR EN UN LOG, output.log, importante, validation también
            # TODO: Incluir también en ese log, el learning rate es interesante ver cómo se va reduciendo
            print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {average_train_loss:.4f}")

            # Por último, validación, SIN updatear los gradientes, solo forward
            self._model.eval()

            validation_accuracy = 0.0
            correct = 0
            total = 0
            epoch_all_predicted_labels = []
            epoch_all_true_labels = []

            # TODO: Cambiar nombres de variables, lo hago por mi pero lo hace un poco engorroso el código.
            # sklearn.metrics.f1_score

            with torch.no_grad():
                for validation_batch_names, validation_batch_labels in dev_batches:
                    validation_batch_names = validation_batch_names.to(self._device)
                    validation_batch_labels = validation_batch_labels.to(self._device)

                    # forward
                    validation_predictions = self._model(validation_batch_names)
                    loss = criterion(validation_predictions, validation_batch_labels)

                    validation_accuracy += loss.item()

                    _, predicted_labels = torch.max(validation_predictions, 1)
                    correct += (predicted_labels == validation_batch_labels).sum().item()
                    total += validation_batch_names.size(0)

                    epoch_all_predicted_labels.extend(predicted_labels.cpu().numpy())
                    epoch_all_true_labels.extend(validation_batch_labels.cpu().numpy())


                average_validation_loss = validation_accuracy / len(dev_batches)
                validation_accuracy = correct / total

                print(f"Epoch {epoch + 1}/{epochs} - Validation Loss: {average_validation_loss:.4f}, "
                      f"Validation Accuracy: {validation_accuracy:.4f}")

            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]

            epoch_stats.append({
                'epoch': epoch + 1,
                'train_loss': average_train_loss,
                'val_loss': average_validation_loss,
                'val_accuracy': validation_accuracy,
                'learning_rate': current_lr
            })

            print(f"Epoch {epoch + 1}/{epochs} - Learning rate adjusted to: {current_lr:.6f}")

            train_losses.append(average_train_loss)
            validation_losses.append(average_validation_loss)

            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy

                indexes, label_str = self._label_encoder.unpack_indexes()
                report = classification_report(epoch_all_true_labels, epoch_all_predicted_labels,
                                               labels=list(indexes),
                                               target_names=list(label_str))

                acc_score = accuracy_score(epoch_all_true_labels, epoch_all_predicted_labels)
                conf_matrix = confusion_matrix(epoch_all_true_labels, epoch_all_predicted_labels)

                best_results = {
                    "accuracy": validation_accuracy,
                    "train_loss": average_train_loss,
                    "val_loss": average_validation_loss,
                    "epoch": epoch + 1,
                    "report": f"Classification Report for Epoch {epoch + 1}:\n{report}\n",
                    "acc_score_sklearn": acc_score,
                    "conf_matrix": conf_matrix
                }

                if save_model:
                    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
                    model_file = f"{output_path}/best_model_{timestamp}.pth"

                    torch.save({
                        'model_state_dict': self._model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': best_validation_accuracy,
                        'epoch': epoch,
                        'char_vocab': self._char_vocab,
                        'label_encoder': self._label_encoder,
                        'hyperparams': hyperparams
                    }, model_file)

                    print(f"Model saved with the best validation accuracy: {best_validation_accuracy:.4f}")

        if log_results:
            self._plot_and_log_results(epoch_stats, epochs, total_train_samples, total_dev_samples, train_losses, validation_losses, hyperparams, best_results, conf_matrix)


    @LoreNexusWrapper._train_mode_only
    def display_batch_info(self, train_batch_names, train_batch_labels, predicted_labels, char_vocab):
        """
        """

        print("Label to Index mapping:", self._label_encoder.label_to_index)
        print("Index to Label mapping:", self._label_encoder.index_to_label)

        # pongo cap de 12, para que no se me vaya de madre el output
        decoded_names = [char_vocab.decode(name)[:12] for name in train_batch_names.tolist()]
        print(f"Name (tokenized and padded): {decoded_names}")

        print(f"Prediction vector (argmax): {predicted_labels}")
        print(f"True label vector         : {train_batch_labels.tolist()}")

        correct_predictions = (predicted_labels == train_batch_labels).sum().item()
        print(f"Correct predictions in this batch: {correct_predictions}/{len(train_batch_labels)}")

        input("Press Enter to continue to the next batch...")


    def load_model(self, model_path='checkpoints/best_model.pth'):
        checkpoint = torch.load(model_path, map_location=self._device, weights_only=False)
        self._char_vocab = checkpoint['char_vocab']
        self._label_encoder = checkpoint['label_encoder']

        hyperparams = checkpoint.get('hyperparams', {})
        embedding_dim = hyperparams.get('embedding_dim', 256)
        hidden_dim = hyperparams.get('hidden_dim', 768)
        num_layers = hyperparams.get('num_layers', 1)
        dropout = hyperparams.get('dropout', 0.0)

        self._hyperparams = hyperparams

        print(f"Loaded Hyperparameters: embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, "
              f"num_layers={num_layers}, dropout={dropout}")


        self._model = self.BiLSTMCharacterLevel(
            vocab_size=len(self._char_vocab),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=len(self._label_encoder),
            n_layers=num_layers,
            dropout=dropout
        ).to(self._device)

        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.eval()


    def predict_name(self, name):
        if self._model is None or self._char_vocab is None:
            raise ValueError("Model is not loaded")

        max_length = 40  # chapuza el hardcodear esto, carga de config.json para todo
        tokenized_name = self._char_vocab.encode(name)
        padded_vector = tokenized_name[:max_length]
        padding_size = max_length - len(padded_vector)

        if padding_size > 0:
            padded_vector.extend([self._char_vocab.get_index(str(SpecialTokens.PAD))] * padding_size)

        input_tensor = tensor([padded_vector], dtype=torch.long).to(self._device)

        with torch.no_grad():
            logits = self._model(input_tensor)

            probabilities = F.softmax(logits, dim=1).squeeze()

            top_indices = torch.argsort(probabilities, descending=True)[:4]

            results = {}
            for i, index in enumerate(top_indices):
                label = self._label_encoder.decode(index.item())
                score = probabilities[index].item()
                results[f"Prediction {i + 1}"] = (label, score)

        return results

    def evaluate(self, verbose=True, model=None):
        """
        """

        if self._mode == "cli_app":
            raise ValueError("Method evaluateis only available in train mode.")

        # TODO: Preparar buen sistema de logging
        if self._model is None:
            if model is not None:
                print("Using provided model for evaluation.")
                self.load_model(model_path=f"checkpoints/{model}")
            else:
                self.load_model()
        else:
            self._model.eval()

        test_split = self._load_data(self._data_folder, splits=['test'])
        test_data = test_split['test']

        test_dataset = self.Dataset(test_data, self._char_vocab)
        test_batches = DataLoader(test_dataset, batch_size=self._hyperparams.get('batch_size', 32), shuffle=False)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_names, batch_labels in test_batches:
                batch_names, batch_labels = batch_names.to(self._device), batch_labels.to(self._device)

                # forward
                outputs = self._model(batch_names)

                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())


        indexes, label_str = self._label_encoder.unpack_indexes()
        report = classification_report(all_labels, all_preds,
                                       labels=list(indexes),
                                       target_names=list(label_str))

        accuracy = accuracy_score(all_labels, all_preds)
        if verbose:
            print("Test set evaluation report:\n", report)
            print(f"Test accuracy: {accuracy:.4f}")
        return accuracy, report

def predict_test(name):

    model = LoreNexusPytorchModel(mode="cli_app")

    results = model.predict_name(name)

    print(f"Prediction results for '{name}':")
    for prediction, (label, score) in results.items():
        print(f"{prediction}: {label} with probability {score:.4f}")


# ln_pytorch_model = LoreNexusPytorchModel(mode='train')
# ln_pytorch_model.train(save_model=True, lr=0.001, epochs=25, hidden_dim=512, embeddings_dim=128, batch_size=64, dropout=0.4, num_layers=2, weight_decay=0.03)
# # ln_pytorch_model.evaluate()