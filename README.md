
<div align="center">
  <h1>LoreNexus</h1>
</div>
<div align="center">
  <img src="images/LoreNexus.png" width="600" alt="LoreNexus">
</div>

---

## Presentación

La presentación del proyecto se puede encontrar aquí: [Presentación de LoreNexus](https://github.com/geru-scotland/lore-nexus/blob/development/doc/LoreNexus-presentacion.pdf)

Memoria en progreso.

---

## 1. Instalación
Para instalar el proyecto y las dependencias, simplemente ejecuta los siguientes comandos:

```bash
git clone git@github.com:geru-scotland/lore-nexus.git
cd lore-nexus
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

---

## 2. Uso del CLI

Para ejecutar la aplicación, navega al modulo `app` y ejecuta `app.py`:

```bash
python3 app/app.py
```

- Si el mejor modelo hasta el momento (`LoreNexusPytorch_v1.0.pth`) no se encuentra, la app lo descarga automáticamente desde Hugging Face:
  [LoreNexusPytorch_v1.0](https://huggingface.co/basajaun-scotland/LoreNexusPytorch_v1.0/tree/main).

---

## 3. Regeneración de datos

El dataset está disponible en `dataset/input`, pero se puede regenerar con  ejecutando `pipeline.py`, simplemente:

   ```bash
   cd pipeline
   python3 pipeline.py
   ```

- La configuración para la regeneración de datos se encuentra en `pipeline/config.json`.
- Los datos ya estratificados se guardan en `dataset/output` (ojo, se sobreescriben los datos existentes)
- También se crea un archivo `data_config.info` que contiene detalles sobre cómo se han generado los datos.

---

## 4. Training Grounds

### Hiperparámetros y experimentos

Para explorar el espacio de hiperparámetros y entrenar modelos, ejecuta `hyperparameter_tuner.py` desde el directorio `training_grounds`:

```bash
cd training_grounds
python3 hyperparameter_tuner.py
```

- Si no se pasan argumentos, se lanzarán experimentos con todos los modelos que implementen la clase `LoreNexusWrapper`, actualmente:


  - `LoreNexusPytorch`
  - `LoreNexusFlair`


- Se pueden pasar los siguientes argumentos para limitar los experimentos a uno de los modelos:

  - `-m pytorch`: Solo entrena el modelo basado en PyTorch.
  - `-m flair`: Solo entrena el modelo basado en Flair.


- Los conjuntos de hiperparámetros para los experimentos están definidos en `param_grids.json` y los logs se guardan en `training_grounds/logs`.

### Entrenar un modelo individualmente

Para entrenar un modelo específico:
1. Ve al archivo correspondiente:
   - `/models/pytorch/model.py`  
   - `/models/flair/model.py`
2. Descomenta las líneas al final del archivo, ajusta los hiperparámetros que desees y ejecutalo:

   ```bash
   python3 models/pytorch/model.py
   ```

   ó

   ```bash
   python3 models/flair/model.py
   ```
   
En ese caso los logs se guardan en `models/pytorch/logs` o `models/flair/logs` respectivamente.

   
No obstante, incluso para entrenamientos individuales, recomiendo usar `hyperparameter_tuner.py`.

---

---

## Ejemplos de predicciones

<div align="center">
  <img src="images/cli-main.png" width="600" alt="CLI Principal">
</div>

---

   <div align="center">
     <img src="images/example-1.png" width="400" alt="Example 1">
   </div>

   <div align="center">
     <img src="images/example-2.png" width="400" alt="Example 2">
   </div>

   <div align="center">
     <img src="images/example-3.png" width="400" alt="Example 3">
   </div>

   <div align="center">
     <img src="images/example-4.png" width="400" alt="Example 4">
   </div>

---

### Tabla de inferencias

<div align="center">
  <img src="images/table-inferences.png" width="600" alt="Table of Inferences">
</div>

---