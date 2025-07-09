# Human-Action-Recognition-System-Using-Deep-Learning
 This project focuses on Human Action Recognition (HAR) using deep learning techniques  to detect and classify actions in both videos and live streams. A hybrid CNN-LSTM model  is used, where CNN extracts spatial features and LSTM captures temporal motion across  frames.

import os
import cv2
import csv
import numpy as np
import time
import threading
from collections import Counter
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

# ============================
# Load the model and class labels
# ============================

def load_action_model(model_path, train_data_path):
    model = load_model(model_path)
    classes = sorted(os.listdir(train_data_path))
    return model, classes

# ============================
# Paths
# ============================

model_path = "human_action_cnn_lstm.h5"
train_data_path = r"C:\Users\Teja\Downloads\HumanActionRecognition\data\preprocessed\train"

model, classes = load_action_model(model_path, train_data_path)

# ============================
# Parameters
# ============================

frames_per_clip = 5
image_size = (224, 224)
delay_between_frames = 2
smoothing_window = 5

# ============================
# Buffers
# ============================

frame_buffer = []
predictions_buffer = []
last_label = ""
last_label_time = 0

# ============================
# ActionApp Class
# ============================

class ActionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Action Recognition")
        self.root.geometry("900x700")

        # Title Label
        self.title_label = tk.Label(root, text="Human Action Recognition",
                                    font=("Helvetica", 20, "bold"))
        self.title_label.pack(pady=10)

        # Video Feed
        self.video_label = tk.Label(root)
        self.video_label.pack()

        # Prediction Text
        self.prediction_text = tk.Label(root, text="Prediction: None",
                                        font=("Helvetica", 16))
        self.prediction_text.pack(pady=10)

        # Confidence Progress Bar
        self.confidence_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(root, orient="horizontal",
                                            length=300, mode="determinate",
                                            variable=self.confidence_var,
                                            maximum=100)
        self.progress_bar.pack(pady=5)

        # Buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=20)

        self.start_button = tk.Button(button_frame, text="▶ Start",
                                      font=("Helvetica", 12), width=10,
                                      command=self.start_recognition)
        self.start_button.grid(row=0, column=0, padx=10)

        self.stop_button = tk.Button(button_frame, text="⬛ Stop",
                                     font=("Helvetica", 12), width=10,
                                     command=self.stop_recognition)
        self.stop_button.grid(row=0, column=1, padx=10)

        self.save_log_button = tk.Button(button_frame, text="💾 Save Log",
                                         font=("Helvetica", 12), width=12,
                                         command=self.save_log)
        self.save_log_button.grid(row=0, column=2, padx=10)

        # Internal State
        self.stop = False
        self.cap = None
        self.log = []

        # CSV Setup
        self.csv_filename = "predictions_log.csv"
        with open(self.csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Predicted Action", "Confidence (%)"])

    def start_recognition(self):
        self.stop = False
        self.log.clear()
        threading.Thread(target=self.video_loop).start()

    def stop_recognition(self):
        self.stop = True

    def save_log(self):
        if not os.path.exists(self.csv_filename):
            messagebox.showinfo("Info", "No log data to save.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv")])
        if save_path:
            try:
                with open(self.csv_filename, "r") as source, open(save_path, "w", newline="") as dest:
                    dest.write(source.read())
                messagebox.showinfo("Saved", f"CSV log saved to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file:\n{e}")

    def video_loop(self):
        global frame_buffer, predictions_buffer, last_label, last_label_time

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            return

        while not self.stop:
            ret, frame = self.cap.read()
            if not ret:
                break

            original_frame = frame.copy()

            processed_frame = cv2.resize(frame, image_size)
            processed_frame = preprocess_input(processed_frame)
            frame_buffer.append(processed_frame)

            for _ in range(delay_between_frames):
                self.cap.read()

            if len(frame_buffer) == frames_per_clip:
                clip = np.array(frame_buffer)
                clip = np.expand_dims(clip, axis=0)

                prediction = model.predict(clip, verbose=0)
                pred_idx = int(np.argmax(prediction))
                predictions_buffer.append(pred_idx)

                if len(predictions_buffer) > smoothing_window:
                    predictions_buffer.pop(0)

                most_common_pred = Counter(predictions_buffer).most_common(1)[0][0]
                confidence = np.max(prediction)
                label = f"{classes[most_common_pred]} ({confidence * 100:.2f}%)"

                last_label = label
                last_label_time = time.time()

                frame_buffer = []

                # UI Update
                self.prediction_text.config(text=f"Prediction: {classes[most_common_pred]}")
                self.confidence_var.set(confidence * 100)

                # Logging
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.log.append(f"[{timestamp}] {label}")

                with open(self.csv_filename, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, classes[most_common_pred], f"{confidence * 100:.2f}"])

            # Overlay prediction text
            if last_label and (time.time() - last_label_time < 2):
                cv2.putText(original_frame, last_label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert to Tkinter Image
            image = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            imgtk = ImageTk.PhotoImage(image=image)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.cap.release()

    def on_close(self):
        self.stop = True
        self.root.destroy()


# ============================
# Run GUI
# ============================

if __name__ == "__main__":
    root = tk.Tk()
    app = ActionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
