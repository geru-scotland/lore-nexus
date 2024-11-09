"""
*****************************************************
 * Universidad del País Vasco (UPV/EHU)
 * Facultad de Informática - Donostia-San Sebastián
 * Asignatura: Procesamiento de Lenguaje Natural
 * Proyecto: Lore Nexus
 *
 * File: stratify_data.py
 * Author: geru-scotland
 * GitHub: https://github.com/geru-scotland
 * Description:
 *****************************************************
"""
import random
from collections import defaultdict

random.seed(42)

with open('data/dataset.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

data_by_label = defaultdict(list)

# Organizo por label, para dividir equitativamente después, si no lo que me ha
# ocurrido es que había alguna label en dev y test que no estaba en train
for line in lines:
    label = line.split()[0]
    data_by_label[label].append(line)

# 80% train, 10% dev/val, 10% test
train_lines, dev_lines, test_lines = [], [], []
for label, items in data_by_label.items():
    random.shuffle(items)

    total_items = len(items)
    train_size = int(0.88 * total_items)
    dev_size = int(0.12 * total_items)
    # test_size = total_items - train_size - dev_size

    train_lines.extend(items[:train_size])
    dev_lines.extend(items[train_size:train_size + dev_size])
    # test_lines.extend(items[train_size + dev_size:])

random.shuffle(train_lines)
random.shuffle(dev_lines)
# random.shuffle(test_lines)

with open('data/train.txt', 'w', encoding='utf-8') as file:
    file.writelines(train_lines)

with open('data/dev.txt', 'w', encoding='utf-8') as file:
    file.writelines(dev_lines)

# with open('data/test.txt', 'w', encoding='utf-8') as file:
#     file.writelines(test_lines)
