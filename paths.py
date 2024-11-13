"""
*****************************************************
 * Universidad del País Vasco (UPV/EHU)
 * Facultad de Informática - Donostia-San Sebastián
 * Asignatura: Procesamiento de Lenguaje Natural
 * Proyecto: Lore Nexus
 *
 * File: paths.py
 * Author: geru-scotland
 * GitHub: https://github.com/geru-scotland
 * Description:
 *****************************************************
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
DATA_OUTPUT_DIR = os.path.join(DATASET_DIR, 'output')
PREPROCESSING_DIR = os.path.join(DATASET_DIR, 'preprocessing')
APIS_DIR = os.path.join(PREPROCESSING_DIR, 'apis')
NER_DIR = os.path.join(PREPROCESSING_DIR, 'ner')

MODELS_DIR = os.path.join(BASE_DIR, 'models')


def get_checkpoints_dir(model_type):
    """
    """
    return os.path.join(MODELS_DIR, model_type, 'checkpoints')