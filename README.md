# 🔥 Detección de Incendios Forestales con RT-DETR

Este repositorio contiene el Trabajo Práctico Final de la materia **Inteligencia Artificial (2025)**.

El objetivo del proyecto es la detección temprana de **Fuego** y **Humo** en entornos forestales utilizando Visión por Computadora. Para ello, implementamos el modelo **RT-DETR (Real-Time Detection Transformer)**, una arquitectura basada en Transformers que ofrece alta precisión y velocidad en tiempo real.

## 📋 Descripción del Proyecto

El sistema es capaz de:
1.  **Entrenar** un modelo personalizado utilizando datasets gestionados en Roboflow.
2.  **Detectar** patrones de fuego y humo en imágenes estáticas.
3.  **Procesar videos** completos para inferencia continua.

### Tecnologías Utilizadas
* **Modelo:** [RT-DETR](https://docs.ultralytics.com/models/rtdetr/) (vía Ultralytics).
* **Gestión de Datos:** Roboflow (Dataset combinado de ~5000 imágenes).
* **Procesamiento:** OpenCV & Supervision.
* **Gestor de Paquetes:** `uv` (para entornos virtuales rápidos).

---

## ⚙️ Instalación y Configuración

Este proyecto utiliza **uv** para la gestión de dependencias y entornos virtuales.

### 1. Prerrequisitos
Tener instalado Python y `uv`. Si no tienes `uv`:
```bash
pip install uv

```

### 2. Clonar repositorio e instalar dependencias

El comando `sync` creará el entorno virtual `.venv` e instalará todo lo necesario.

**Para uso con GPU (Recomendado):**

```bash
uv sync --extra gpu

```

**Para uso con CPU:**

```bash
uv sync --extra cpu

```

### 3. Configurar Variables de Entorno

Para entrenar, necesitas acceso a Roboflow. Crea un archivo `.env` en la raíz del proyecto y agrega tu API Key:

```env
ROBOFLOW_API_KEY=tu_clave_privada_aqui

```

---

## 🚀 Entrenamiento del Modelo (`train.py`)

El script `train.py` descarga automáticamente la última versión del dataset desde Roboflow y comienza el fine-tuning del modelo `rtdetr-l.pt`.

**Ejecutar entrenamiento:**

Con GPU (NVIDIA):

```bash
uv run --extra gpu python train.py

```

Con CPU (Lento, solo para pruebas):

```bash
uv run --extra cpu python train.py

```

> **Nota:** Por defecto, el script está configurado en `epochs=1` para verificar que todo funcione. Para un entrenamiento real, edita `train.py` y cambia a `epochs=100`.

---

## 👁️ Inferencia y Pruebas

Una vez que tengas tu modelo entrenado (el archivo `best.pt`), colócalo en la carpeta raíz del proyecto para realizar pruebas.

### 1. Detección en Imagen (`run.py`)

Analiza una imagen estática (`imagen_prueba.jpg`) y guarda el resultado visual con las cajas delimitadoras.

```bash
uv run --extra gpu python run.py

```

* **Entrada:** `imagen_prueba.jpg`
* **Salida:** `resultado_rfdetr.jpg`

### 2. Detección en Video (`run_video.py`)

Procesa un video completo frame por frame. Al ejecutarlo, te pedirá el nombre del archivo de video.

```bash
uv run --extra gpu python run_video.py

```

* **Entrada:** Te pedirá el nombre (ej: `video_fuego.mp4`).
* **Salida:** `video_resultado.mp4`
