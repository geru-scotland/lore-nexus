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
from enum import Enum
from pathlib import Path

import torch.utils.data

sys.path.append(str(Path(__file__).resolve().parents[2]))

from models.lorenexus.lorenexus import LoreNexusWrapper

from torch.utils.data import DataLoader


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
            self.char2index[SpecialTokens.PAD] = 0
            self.index2char[0] = SpecialTokens.PAD
            self.char2index[SpecialTokens.UNK] = 1
            self.index2char[1] = SpecialTokens.UNK

        def encode(self, word):
            tokenized_chars = []
            for char in word:
                if char not in self.char2index:
                    index = len(self)
                    self.char2index[char] = index
                    self.index2char[index] = char
                    tokenized_chars.append(index)
            return tokenized_chars

        def get_index(self, char=None):
            return self.char2index.get(char, self.char2index[SpecialTokens.UNK])

        def get_char(self, index=None):
            return self.index2char.get(index, SpecialTokens.UNK)

        def decode(self, indices):
            # Implemento decode porque en un futuro intentaré generar nombres
            return "".join([self.index2char[index] for index in indices if index in self.index2char])

        def build_vocab(self, data):
            self._add_special_tokens()
            for word in data:
                self.encode(word)


    class Dataset(torch.utils.data.Dataset):
        def __init__(self, data, char_vocab, max_length=20):
            self._max_length = max_length
            self._data = data
            self._char_vocab = char_vocab
            self.tokenize_data()

        def __len__(self):
            return len(self._data[0])

        def __getitem__(self, index):
            """
            Tengo que devolver el nombre tokenizado y label, esto lo llamará el DataLoader
            para pasar inputs y targets al modelo.
            Tokenizo el nombre y lo corto si es más largo que el máximo
            TODO: Mirar como determinar el máximo, ahora pongo 20 a lo loco
            """
            names, labels = self._data[index]
            name, label = names[index], labels[index]

            tokenized_name = self._char_vocab.encode(name)
            padded_vector = tokenized_name[:self._max_length] # corto, si faltan va a haber que rellenar
            padding_size = self._max_length - len(padded_vector)

            if padding_size > 0:
                padded_vector.extend([self._char_vocab.get_index(SpecialTokens.PAD)]*padding_size)

            return padded_vector, label

        def tokenize_data(self):
            self._char_vocab.build_vocab(self._data)

    class BiLSTMCharacterLevel(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
            super().__init__()

            # En este caso, no es necesario utilizar nn.Parameters, no utilizo
            # tensores custom, estos los gestiona internamente PyTorch
            self.character_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

            self.bi_lstm = torch.nn.LSTM(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=n_layers,
                bidirectional=True,
                batch_first=True, # por compatibilidad, con shape (batch_size, sequence_len, embedding_dim)
                dropout=dropout)

            # OJO! bidireccional, por eso el *2 en el hidden_dim, uno para cada dirección
            self.fc = torch.nn.Linear(hidden_dim*2, output_dim)
        def forward(self, input_tensor):
            # input_tensor shape: (batch_size, sequence_len)
            # Ejemplo que me ayuda a verlo:
            # input_tensor = torch.tensor([
            #     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 3, 12, 4, 11, 0],  # "Luke Skywalker" + padding
            #     [6, 2, 11, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # "Sue" + padding
            # ])
            embedded_tensor = self.character_embeddings(input_tensor)
            # embedded_tensor shape: (batch_size, sequence_len, embedding_dim)
            # Cada capa 2D del stack, tiene tantas filas como caracteres en el nombre completo
            # y tantas columnas como dimensiones en el embedding. Y tantas capas como palabras.

            # colección de los estados ocultos intermedios
            # hidden state es el final
            # cell state, el estado de la celda lstm
            lstm_out, (hidden_state, cell) = self.bi_lstm(embedded_tensor)

            # Shape de lstm_out: (batch_size, max_length, hidden_dim * 2)
            # Shape de hidden: (batch_size, num_layers * num_directions (2), hidden_dim)

            # Imagina (50 nombres por batch, 1 capa * 2 direcciones, 20 de hidden_dim)
            # si concatenamos por la dim=1, es por el numero de estados hidden, 1 por direccion (si una capa). Así que
            # si teniamos 50 channels stackeados de 2x20, ahora tendremos 1 channel de 50*2x20

            # TODO: Revisar bien esta concatenación para comprender bien
            # Tenemos 2 conjuntos de estados ocultos, uno para izquierda y otro para derecha
            # La idea es concatenar los estados ocultos de cada dirección
            # como es batch first, shape es (batch_size, num_layers * num_directions, hidden_dim)
            # y si concateno por dim 2 el resultado es (batch_size, hidden_dim * 2)
            # lo uqe me interesa es unir o concatenar los dos vectores en uno solo, y eso se hace uniendo columnas

            # Básicamente estoy concatenando los estados ocultos finales de las dos direcciones
            hidden = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=2)

            out = self.fc(hidden)

            return out

    def __init__(self, mode="train", data_folder='data/', model_path=None):
        """
        """
        super().__init__(mode)

        self._mode = mode
        self._data_folder = data_folder

    def _load_data(self, data_folder):
        """
        """
        data = {}
        for split in ['train', 'dev', 'test']:
            file_path = Path(data_folder) / f"{split}.txt"
            names, labels = [], []

            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()

                    parts = line.split(' ', 1)
                    label = parts[0].replace('__label__', '')
                    name = parts[1]
                    names.append(name)
                    labels.append(label)

            data[split] = (names, labels)

        return data['train'], data['dev'], data['test']

    @LoreNexusWrapper._train_mode_only
    def train(self, output_path='', lr=0.001, batch_size=32, epochs=10):
        """
        TODO: Cargar del config.json
        """
        # 1) Creo el vocabulario
        # 2) Cargo los datos
        # 3) Creo el modelo
        # 4) Entreno
        char_vocab = self.CharacterVocab()
        train_data, dev_data, test_data = self._load_data(self._data_folder)

        train_dataset = self.Dataset(train_data, char_vocab)
        dev_dataset = self.Dataset(dev_data, char_vocab)
        test_dataset = self.Dataset(test_data, char_vocab)

        # Me devuelve iterables, que son los batches
        train_batches = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_batches = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
        tes_batches = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Creo el modelo
        self._model = self.BiLSTMCharacterLevel(
            vocab_size=len(char_vocab),
            embedding_dim=256,
            hidden_dim=256,
            output_dim=5,
            n_layers=1,
            dropout=0.2
        ).to(self._device)

        weight_decay = 0.01
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self._model.parameters(), weight_decay=weight_decay, lr=lr)

        # TODO: Como tengo clases un poco desbalanceadas, utilizaré métrica F1 para que
        # mida el rendimiento un poco más completo

        best_validation_loss = float("inf")

        for epoch in range(epochs):
            self._model.train()
            train_loss = 0.0
            for train_batch_names, train_batch_labels in train_batches:
                train_batch_names = train_batch_names.to(self._device)
                train_batch_labels = train_batch_labels.to(self._device)

                # forward
                optimizer.zero_grad() # siempre hay que resetear gradientes después de un forward
                train_predictions = self._model(train_batch_names) # forward
                loss = criterion(train_predictions, train_batch_labels)
                train_loss += loss.item()

                # backward
                loss.backward() # solo calcula los gradientes
                optimizer.step() # y esto ya, updatea pesos

            # Logística para calculo de pérdida en cada epoch, de entre todos los batches
            average_train_loss = train_loss / len(train_batches)

            # TODO: mejorar el output
            print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {average_train_loss:.4f}")

            # Por último, validación, SIN updatear los gradientes, solo forward

            self._model.eval()

            validation_loss = 0.0
            correct = 0
            total = 0
            # TODO: Cambiar nombres de variables, lo hago por mi pero lo hace un poco engorroso el código.
            # sklearn.metrics.f1_score
            with torch.no_grad():
                for validation_batch_names, validation_batch_labels in dev_batches:
                    validation_batch_names = validation_batch_names.to(self._device)
                    validation_batch_labels = validation_batch_labels.to(self._device)

                    # forward
                    validation_predictions = self._model(validation_batch_names)
                    loss = criterion(validation_predictions, validation_batch_labels)

                    validation_loss += loss.item()

                    # validation_predictions es un tensor, con los logits sin normalizar
                    # max coge el máximo, para poder saber cual es la clase predicham dimension 1 porque
                    # la cero es cada instancia y las características todas las labels
                    # Nombre StarWars  LOTR  GOT
                    # nedstark  1.2     2.7  3.5
                    _, predicted_labels = torch.max(validation_predictions, 1)
                    correct += (predicted_labels == validation_batch_labels).sum().item() # conteo de correctas
                    total += validation_batch_names.size(0) # tamaño del batch

                average_validation_loss = validation_loss / len(dev_batches)
                validation_accuracy = correct / total
                print(f"Epoch {epoch + 1}/{epochs} - Validation Loss: {average_validation_loss:.4f}, "
                      f"Validation Accuracy: {validation_accuracy:.4f}")

            if average_validation_loss < best_validation_loss:
                best_validation_loss = average_validation_loss
                torch.save(self._model.state_dict(), f"checkpoints/best_model.pth")
                print(f"Model saved with the best validation loss: {best_validation_loss}")



    def evaluate(self, eval_data):
        """
        """
        pass

    def predict_name(self, name):
        """
        """
        pass
