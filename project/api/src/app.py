import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
import os
import sys
import imutils
import numpy as np
import threading

def _base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

BASE_DIR = _base_dir()
DATA_PATH = os.path.join(BASE_DIR, 'Fotos')
MODEL_PATH = os.path.join(BASE_DIR, 'Modelos')
MODEL_FILE = os.path.join(MODEL_PATH, 'modeloLBPHFace.xml')


def log(text_widget, message):
    text_widget.configure(state='normal')
    text_widget.insert(tk.END, message + '\n')
    text_widget.see(tk.END)
    text_widget.configure(state='disabled')


def capturar(nombre, log_widget):
    if not nombre.strip():
        messagebox.showwarning("Nombre vacío", "Ingresa un nombre para la persona.")
        return

    person_path = os.path.join(DATA_PATH, nombre.strip())
    if not os.path.exists(person_path):
        os.makedirs(person_path)
        log(log_widget, f"Carpeta creada: {person_path}")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    face_classif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0
    log(log_widget, f"Capturando fotos de '{nombre.strip()}'... Presiona ESC para detener.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aux_frame = frame.copy()
        faces = face_classif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rostro = aux_frame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(person_path, f'rostro_{count}.jpg'), rostro)
            count += 1

        cv2.putText(frame, f'Fotos: {count}/100', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow('Capturando - ESC para salir', frame)

        k = cv2.waitKey(1)
        if k == 27 or count >= 100:
            break

    cap.release()
    cv2.destroyAllWindows()
    log(log_widget, f"Captura terminada. Total de fotos guardadas: {count}")


def entrenar(log_widget):
    if not os.path.exists(DATA_PATH) or len(os.listdir(DATA_PATH)) == 0:
        messagebox.showerror("Sin datos", f"No hay fotos en {DATA_PATH}.\nPrimero captura personas.")
        return

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    people_list = os.listdir(DATA_PATH)
    log(log_widget, f"Personas encontradas: {people_list}")

    labels = []
    faces_data = []
    label = 0

    for name_dir in people_list:
        person_path = os.path.join(DATA_PATH, name_dir)
        log(log_widget, f"Leyendo imágenes de: {name_dir}")
        for file_name in os.listdir(person_path):
            img_path = os.path.join(person_path, file_name)
            img = cv2.imread(img_path, 0)
            if img is not None:
                faces_data.append(img)
                labels.append(label)
        label += 1

    log(log_widget, "Entrenando modelo LBPH...")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces_data, np.array(labels))
    face_recognizer.write(MODEL_FILE)
    log(log_widget, f"Modelo guardado en: {MODEL_FILE}")
    messagebox.showinfo("Entrenamiento", "¡Modelo entrenado y guardado correctamente!")


def reconocer(log_widget):
    if not os.path.exists(MODEL_FILE):
        messagebox.showerror("Sin modelo", "No se encontró el modelo entrenado.\nPrimero ejecuta el entrenamiento.")
        return

    people_list = os.listdir(DATA_PATH)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)
    face_classif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    log(log_widget, "Reconocimiento en tiempo real iniciado. Presiona ESC para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aux_frame = gray.copy()
        faces = face_classif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            rostro = aux_frame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            result = recognizer.predict(rostro)

            cv2.putText(frame, f'({result[0]}, {result[1]:.1f})', (x, y - 5), 1, 1.1, (255, 255, 0), 1, cv2.LINE_AA)

            if result[1] < 85:
                nombre = people_list[result[0]] if result[0] < len(people_list) else "?"
                cv2.putText(frame, nombre, (x, y - 25), 2, 1.0, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            elif result[1] <= 110:
                cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow('Reconocimiento - ESC para salir', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    log(log_widget, "Reconocimiento finalizado.")


def run_in_thread(fn, *args):
    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()


def build_gui():
    root = tk.Tk()
    root.title("Sistema de Reconocimiento Facial")
    root.geometry("600x520")
    root.resizable(False, False)
    root.configure(bg="#1e1e2e")

    # ── Título ──────────────────────────────────────────────────────────────
    title = tk.Label(root, text="Reconocimiento Facial", font=("Helvetica", 18, "bold"),
                     bg="#1e1e2e", fg="#cdd6f4")
    title.pack(pady=(18, 4))

    subtitle = tk.Label(root, text="Captura · Entrena · Reconoce",
                        font=("Helvetica", 10), bg="#1e1e2e", fg="#6c7086")
    subtitle.pack(pady=(0, 14))

    # ── Frame de nombre ──────────────────────────────────────────────────────
    name_frame = tk.Frame(root, bg="#1e1e2e")
    name_frame.pack(pady=(0, 10))

    tk.Label(name_frame, text="Nombre de persona:", bg="#1e1e2e", fg="#cdd6f4",
             font=("Helvetica", 10)).grid(row=0, column=0, padx=(0, 8))
    name_var = tk.StringVar()
    name_entry = tk.Entry(name_frame, textvariable=name_var, width=22,
                          font=("Helvetica", 11), bg="#313244", fg="#cdd6f4",
                          insertbackground="#cdd6f4", relief="flat", bd=4)
    name_entry.grid(row=0, column=1)

    # ── Botones ──────────────────────────────────────────────────────────────
    btn_frame = tk.Frame(root, bg="#1e1e2e")
    btn_frame.pack(pady=8)

    BTN_STYLE = {
        "font": ("Helvetica", 11, "bold"),
        "width": 18,
        "height": 2,
        "relief": "flat",
        "cursor": "hand2",
        "bd": 0,
    }

    # Log widget (se pasa como argumento a los callbacks)
    log_frame = tk.Frame(root, bg="#1e1e2e")
    log_frame.pack(fill="both", expand=True, padx=16, pady=(8, 16))

    tk.Label(log_frame, text="Registro de actividad", bg="#1e1e2e", fg="#6c7086",
             font=("Helvetica", 9)).pack(anchor="w")

    log_text = tk.Text(log_frame, height=11, bg="#181825", fg="#a6e3a1",
                       font=("Consolas", 9), relief="flat", state='disabled',
                       wrap="word")
    log_text.pack(fill="both", expand=True)

    scrollbar = ttk.Scrollbar(log_frame, command=log_text.yview)
    log_text.configure(yscrollcommand=scrollbar.set)

    # Definir botones ahora que log_text existe
    btn_capture = tk.Button(btn_frame, text="📷  Capturar", bg="#89b4fa", fg="#1e1e2e",
                            command=lambda: run_in_thread(capturar, name_var.get(), log_text),
                            **BTN_STYLE)
    btn_capture.grid(row=0, column=0, padx=8, pady=6)

    btn_train = tk.Button(btn_frame, text="🧠  Entrenar", bg="#a6e3a1", fg="#1e1e2e",
                          command=lambda: run_in_thread(entrenar, log_text),
                          **BTN_STYLE)
    btn_train.grid(row=0, column=1, padx=8, pady=6)

    btn_recognize = tk.Button(btn_frame, text="🔍  Reconocer", bg="#fab387", fg="#1e1e2e",
                              command=lambda: run_in_thread(reconocer, log_text),
                              **BTN_STYLE)
    btn_recognize.grid(row=0, column=2, padx=8, pady=6)

    # Estado del modelo
    def refresh_status():
        if os.path.exists(MODEL_FILE):
            people = os.listdir(DATA_PATH) if os.path.exists(DATA_PATH) else []
            status_var.set(f"✅ Modelo listo  |  Personas: {', '.join(people) if people else 'ninguna'}")
            status_label.config(fg="#a6e3a1")
        else:
            status_var.set("⚠️  Sin modelo entrenado")
            status_label.config(fg="#f38ba8")
        root.after(3000, refresh_status)

    status_var = tk.StringVar()
    status_label = tk.Label(root, textvariable=status_var, bg="#1e1e2e",
                            font=("Helvetica", 9))
    status_label.pack(pady=(0, 6))
    refresh_status()

    log(log_text, "Sistema iniciado. Listo para usar.")

    root.mainloop()


if __name__ == '__main__':
    build_gui()
