# Audio Event Detector

Sistema de detección automática de eventos sonoros utilizando redes neuronales convolucionales (CNN) y procesamiento digital de audio.

El proyecto fue desarrollado para clasificar y detectar eventos acústicos a partir de archivos de audio, transformando señales temporales en representaciones espectrales aptas para modelos de machine learning.

## Características

* Procesamiento automático de archivos de audio.
* Conversión de audio a Mel Spectrograms.
* Entrenamiento de modelos de deep learning para clasificación sonora.
* Pipeline completo de preprocessing, entrenamiento y evaluación.
* Visualización de espectrogramas.
* Inferencia sobre nuevos audios.
* Evaluación de métricas de desempeño.

## Tecnologías utilizadas

* Python
* PyTorch
* NumPy
* pandas
* librosa
* matplotlib
* scikit-learn

## Objetivo del proyecto

El objetivo principal del proyecto es detectar y clasificar eventos acústicos mediante técnicas modernas de machine learning y análisis de señales.

El sistema transforma señales de audio crudas en representaciones frecuenciales utilizando Mel Spectrograms, que luego son procesados por una red neuronal convolucional entrenada para reconocer patrones acústicos específicos.

## Pipeline del sistema

```text
Audio (.wav)
        ↓
Preprocessing
        ↓
Mel Spectrogram
        ↓
Normalización
        ↓
CNN
        ↓
Clasificación
```

## Procesamiento de audio

El pipeline de preprocessing incluye:

* Carga de archivos de audio.
* Conversión de sample rate.
* Generación de espectrogramas.
* Conversión a escala Mel.
* Normalización de datos.
* Preparación de batches para entrenamiento.

## Dataset

El proyecto utiliza una estructura organizada de datasets crudos, procesados y externos.

Actualmente se trabaja con el dataset ESC-50, ampliamente utilizado para clasificación de eventos acústicos ambientales.

La estructura separa:

* Datos originales (`raw`).
* Datos transformados (`processed`).
* Recursos externos (`external`).
* Tests y validaciones.

## Modelo

El modelo principal utiliza una arquitectura CNN para aprender patrones espaciales sobre espectrogramas de audio.

La red convolucional permite detectar estructuras temporales y frecuenciales relevantes para la clasificación de eventos acústicos.

## Estructura general

````text
project/
├── data
│   ├── external
│   ├── processed
│   └── raw
│       └── esc50
│           ├── audio
│           ├── meta
│           └── tests
├── models
│   ├── checkpoints
│   └── tflite
├── notebooks
├── scripts
└── src
    ├── audio_event_detector
    │   ├── dataset
    │   ├── export
    │   ├── features
    │   ├── inference
    │   └── models
    └── audio_event_detector.egg-info
```text
project/
├── data/
├── models/
├── notebooks/
├── preprocessing.py
├── train.py
├── predict.py
├── model.py
├── utils.py
└── README.md
````

## Instalación

Clonar el repositorio:

```bash
git clone https://github.com/usuario/audio-event-detector.git
cd audio-event-detector
```

Crear entorno virtual:

```bash
python -m venv venv
source venv/bin/activate
```

Instalar dependencias:

```bash
pip install -e .
```

El proyecto utiliza `pyproject.toml` para manejo de dependencias y empaquetado.

## Entrenamiento

Ejecutar entrenamiento:

```bash
python train.py
```

## Inferencia

Realizar predicciones sobre nuevos audios:

```bash
python predict.py --input audio.wav
```

## Visualización de espectrogramas

Ejemplo conceptual de transformación de audio:

```text
Waveform → FFT → Mel Scale → Spectrogram
```

## Evaluación

El proyecto incluye evaluación mediante métricas de clasificación como:

* Accuracy
* Precision
* Recall
* F1-score

## Aplicaciones posibles

* Monitoreo acústico.
* Detección de eventos industriales.
* Clasificación ambiental.
* Sistemas de vigilancia acústica.
* Audio tagging.
* Sistemas inteligentes basados en sonido.

## Desarrollo del proyecto

El proyecto fue desarrollado como práctica aplicada de machine learning y procesamiento digital de señales.

Durante el desarrollo se trabajó sobre:

* Procesamiento de audio.
* Transformadas espectrales.
* Redes neuronales convolucionales.
* Pipelines de entrenamiento.
* Evaluación de modelos.
* Manipulación de datasets acústicos.
* Visualización de datos.

## Posibles mejoras futuras

* Inferencia en tiempo real.
* Arquitecturas más avanzadas.
* Data augmentation.
* Entrenamiento distribuido.
* Deployment como API.
* Interfaz web.
* Transfer learning.

## Licencia

El código publicado en este repositorio tiene fines educativos.
