# 🐜 TFG – Desarrollo de un sistema de visión artificial
para el estudio del comportamiento de hormigas

Repositorio del Trabajo Fin de Grado de Alejandro Peñafiel Pujante.  
Incluye scripts de detección, autoetiquetado, modificación de datasets, modelos entrenados y análisis estadístico.

---

## 📂 Estructura del repositorio

```text
scripts/
├── ant_detection/
│   ├── apply_bgsub_tracker.py
│   └── bgsub_ant_tracker.py
│
├── autolabel/
│   ├── autolabel.py
│   └── processor.py
│
└── dataset_modification/
    └── modify_dataset.py

models/
└── ant_yolov8x.pt

R_analysis/
├── bgsub_fine_tuning.Rmd
├── ant_videos_labels.csv
├── ant_videos_labels_white.csv
├── grid_results_20260504_091637.csv
└── grid_results_20260518_194907.csv

TFG_Alejandro_Penafiel_Pujante.pdf
```

---

## 🐜 `scripts/ant_detection/`

### **`apply_bgsub_tracker.py`**
Script para realizar la detección mediante sustracción de fondo sobre **una carpeta completa de vídeos**.

### **`bgsub_ant_tracker.py`**
Módulo que implementa el **tracking de hormigas en un único vídeo**, utilizando background subtraction y filtrado morfológico.

---

## 🏷️ `scripts/autolabel/`

### **`autolabel`**
Script que **autoetiqueta un vídeo**, generando cada *X* frames imágenes y etiquetas en formato YOLO.

### **`processor`**
Módulo que implementa diferentes métodos de autoetiquetado:
- Sustracción de fondo  
- Flujo óptico denso  
- Flujo óptico discreto  
- Diferencias de bordes (Canny)

---

## 🛠️ `scripts/dataset_modification/`

### **`modify_dataset.py`**
Script generado con Copilot para **modificar datasets YOLO**, permitiendo eliminar, añadir o modificar bounding boxes, con hasta un máximo de 10 clases.

---

## 🧠 `models/`

### **`ant_yolov8x.pt`**
Modelo YOLOv8x entrenado para la **detección de hormigas en vídeos de alta calidad**.

---

## 📊 `R_analysis/`

### **`bgsub_fine_tuning.Rmd`**
Documento R Markdown con el **análisis estadístico** utilizado para seleccionar la mejor combinación de hiperparámetros del método de sustracción de fondo.

### **`ficheros csv`**
Ficheros csv con los datos utilizados en el análisis.

---

## 📄 Documento del TFG

### **`TFG_Alejandro_Penafiel_Pujante.pdf`**
Memoria completa del Trabajo Fin de Grado, describiendo metodología, experimentos, resultados y conclusiones.

 
