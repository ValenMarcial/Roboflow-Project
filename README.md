# Roboflow-Project
Este proyecto es un trabajo practico de la materia Inteligencia Artificial, basado en el uso de modelos de roboflow que estan implementados con Transformers (RF-DETR)

# Instalacion de dependencias
## uv
```bash
  pip install uv
```

# Formas de correr con uv el entrenamiento

### Esto usa el índice pytorch-cu126
```sh
  uv run --extra gpu train.py
```

### Esto usa el índice pytorch-cu126
```sh
  uv run --extra cpu train.py
```

# Formas de correr con uv el uso del modelo

### Esto usa el índice pytorch-cpu
```bash
  uv run --extra cpu run.py
```

### Esto usa el índice pytorch-cpu
```sh
  uv run --extra gpu run.py
```

# Si queremos que uv nos cree el propio venv

### Para GPU
```sh
  uv sync --extra gpu
```

### Para CPU
```sh
  uv sync --extra cpu
```

### para activar y correr
```sh
  source .venv/bin/activate
  python train.py
  python run.py
```