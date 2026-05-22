# 🎭 Sistema de Reconocimiento Facial

Sistema de reconocimiento facial en tiempo real desarrollado en Python, utilizando OpenCV con el algoritmo **LBPH (Local Binary Pattern Histograms)** y una interfaz gráfica de escritorio construida con **Tkinter**.

---

## 📋 Descripción

Este proyecto implementa un pipeline completo de reconocimiento facial:

1. **Captura** — Toma hasta 100 fotografías del rostro de una persona usando la cámara web.
2. **Entrenamiento** — Entrena un modelo LBPH con las fotos capturadas de todas las personas registradas.
3. **Reconocimiento** — Identifica en tiempo real a las personas registradas a través de la cámara.

Todo integrado en una interfaz gráfica simple y sin necesidad de usar la terminal.

---

## 🗂️ Estructura del Proyecto

```
Proyecto-reconocimiento-facial/
├── project/
│   └── api/
│       ├── src/
│       │   ├── app.py               # Interfaz gráfica principal (punto de entrada)
│       │   ├── capturando.py        # Script standalone de captura
│       │   ├── entrenador.py        # Script standalone de entrenamiento
│       │   ├── reconocimiento.py    # Script standalone de reconocimiento
│       │   └── prueba.py            # Verificación del entorno
│       ├── Fotos/                   # Generado automáticamente al capturar
│       └── Modelos/                 # Generado automáticamente al entrenar
├── .gitignore
└── README.md
```

---

## 🚀 Instalación y Uso

### Requisitos

- Python 3.7 o superior
- Cámara web

### Instalar dependencias

```bash
pip install opencv-contrib-python imutils numpy
```

> ⚠️ Usar `opencv-contrib-python` (no `opencv-python`) ya que se requiere el módulo `cv2.face`.

### Ejecutar la aplicación

```bash
python project/api/src/app.py
```

### Flujo de uso

| Paso | Acción | Descripción |
|------|--------|-------------|
| 1 | Escribir nombre | Ingresar el nombre de la persona en el campo de texto |
| 2 | **Capturar** | Abre la cámara y guarda 100 fotos del rostro |
| 3 | **Entrenar** | Genera el modelo con todas las personas registradas |
| 4 | **Reconocer** | Inicia reconocimiento facial en tiempo real |

---

## 📦 Generar ejecutable `.exe`

Para distribuir el sistema sin necesidad de instalar Python:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name "ReconocimientoFacial" project/api/src/app.py
```

El ejecutable se genera en la carpeta `dist/`. Las carpetas `Fotos/` y `Modelos/` se crean automáticamente junto al `.exe` al ejecutarlo.

---

## 🧠 Tecnologías utilizadas

| Tecnología | Uso |
|------------|-----|
| [OpenCV](https://opencv.org/) | Detección y reconocimiento facial |
| LBPH Face Recognizer | Algoritmo de reconocimiento |
| Haar Cascade | Detector de rostros frontal |
| Tkinter | Interfaz gráfica de escritorio |
| imutils | Utilidades de procesamiento de imagen |
| NumPy | Manejo de arrays para el modelo |

---

## ⚙️ Parámetros del reconocedor

| Confianza | Resultado |
|-----------|-----------|
| `< 85` | ✅ Persona reconocida (recuadro verde) |
| `85 – 110` | ⚠️ Desconocido (recuadro rojo) |
| `> 110` | No se muestra etiqueta |

> Mientras **menor** sea el valor de confianza, **mayor** es la certeza del reconocimiento.

---

## 👤 Autor

Desarrollado como proyecto de avance académico.
