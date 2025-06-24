import os
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk


class ImageProcessorApp:
    """Графическое приложение для обработки изображений."""

    def __init__(self, root):
        """Инициализация приложения."""
        self.root = root
        self.root.title(f"Работа с изображениями")
        self.root.geometry("1000x700")

        # Инициализация переменных
        self.image = None
        self.original_image = None
        self.tk_image = None
        self.webcam = None
        self.webcam_active = False

        # Создание интерфейса
        self.create_widgets()

    def create_widgets(self):
        """Создание элементов интерфейса."""
        # Основные фреймы
        self.image_frame = ttk.Frame(self.root)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.control_frame = ttk.Frame(self.root, width=300)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # Холст для изображения
        self.canvas = tk.Canvas(self.image_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Элементы управления
        ttk.Button(
            self.control_frame,
            text="Загрузить изображение",
            command=self.load_image
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            self.control_frame,
            text="Сделать фото с камеры",
            command=self.toggle_webcam
        ).pack(fill=tk.X, pady=2)

        # Выбор каналов
        self.channel_var = tk.StringVar(value="RGB")
        ttk.Label(self.control_frame, text="Цветовые каналы:").pack(pady=(10, 0))

        channels = [
            ("RGB", "RGB"),
            ("Красный", "R"),
            ("Зеленый", "G"),
            ("Синий", "B")
        ]

        for text, value in channels:
            ttk.Radiobutton(
                self.control_frame,
                text=text,
                variable=self.channel_var,
                value=value,
                command=self.update_channels
            ).pack(anchor=tk.W)

        # Дополнительные функции
        ttk.Label(
            self.control_frame,
            text="Дополнительные функции:"
        ).pack(pady=(10, 0))

        ttk.Button(
            self.control_frame,
            text="Негатив изображения",
            command=self.apply_negative
        ).pack(fill=tk.X, pady=2)

        # Вращение
        ttk.Label(self.control_frame, text="Угол вращения:").pack(pady=(5, 0))
        self.rotate_entry = ttk.Entry(self.control_frame)
        self.rotate_entry.pack(fill=tk.X, pady=2)

        ttk.Button(
            self.control_frame,
            text="Повернуть изображение",
            command=self.rotate_image
        ).pack(fill=tk.X, pady=2)

        # Рисование круга
        ttk.Label(self.control_frame, text="Координаты круга:").pack(pady=(5, 0))
        
        # Фрейм для координат X и Y
        coords_frame = ttk.Frame(self.control_frame)
        coords_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(coords_frame, text="X:").pack(side=tk.LEFT, padx=(0, 5))
        self.circle_x_entry = ttk.Entry(coords_frame, width=10)
        self.circle_x_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(coords_frame, text="Y:").pack(side=tk.LEFT, padx=(5, 5))
        self.circle_y_entry = ttk.Entry(coords_frame, width=10)
        self.circle_y_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(self.control_frame, text="Радиус круга:").pack(pady=(5, 0))
        self.circle_radius_entry = ttk.Entry(self.control_frame)
        self.circle_radius_entry.pack(fill=tk.X, pady=2)

        ttk.Button(
            self.control_frame,
            text="Нарисовать круг",
            command=self.draw_circle
        ).pack(fill=tk.X, pady=2)

        # Сброс
        ttk.Button(
            self.control_frame,
            text="Сбросить изменения",
            command=self.reset_image
        ).pack(fill=tk.X, pady=(10, 2))

    def load_image(self):
        """Загрузка изображения из файла с поддержкой кириллических путей."""
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("JPEG images", "*.jpg *.jpeg"),
                ("PNG images", "*.png"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Файл не найден: {file_path}")

            ext = os.path.splitext(file_path)[1].lower()
            if ext not in (".jpg", ".jpeg", ".png"):
                raise ValueError("Поддерживаются только файлы JPG/JPEG/PNG")

            # Используем numpy для чтения файла с кириллическими путями
            with open(file_path, 'rb') as f:
                file_bytes = np.frombuffer(f.read(), np.uint8)
                self.original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if self.original_image is None:
                raise ValueError("OpenCV не смог загрузить изображение")

            # Конвертируем в PyTorch tensor и обратно для демонстрации
            tensor_image = torch.from_numpy(self.original_image).float() / 255.0
            self.original_image = (tensor_image.numpy() * 255).astype(np.uint8)
            
            self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.show_image()

        except Exception as e:
            error_msg = f"Не удалось загрузить изображение:\n{str(e)}"
            messagebox.showerror("Ошибка загрузки", error_msg)
            print(f"Ошибка загрузки: {traceback.format_exc()}")

    def toggle_webcam(self):
        """Включение/выключение веб-камеры."""
        if self.webcam_active:
            self.stop_webcam()
        else:
            self.start_webcam()

    def start_webcam(self):
        """Запуск веб-камеры."""
        try:
            self.webcam = cv2.VideoCapture(0)
            if not self.webcam.isOpened():
                raise ValueError("Не удалось подключиться к камере")

            self.webcam_active = True
            self.control_frame.winfo_children()[1].config(text="Сделать снимок")
            self.capture_from_webcam()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка веб-камеры: {str(e)}")
            self.webcam_active = False
            if self.webcam is not None:
                self.webcam.release()
            self.webcam = None

    def stop_webcam(self):
        """Остановка веб-камеры."""
        self.webcam_active = False
        self.control_frame.winfo_children()[1].config(text="Сделать фото с камеры")
        if self.webcam is not None:
            self.webcam.release()
            self.webcam = None

    def capture_from_webcam(self):
        """Захват кадров с веб-камеры."""
        if not self.webcam_active:
            return

        ret, frame = self.webcam.read()
        if ret:
            # Конвертируем в PyTorch tensor и обратно для демонстрации
            tensor_frame = torch.from_numpy(frame).float() / 255.0
            self.original_image = (tensor_frame.numpy() * 255).astype(np.uint8)
            
            self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.show_image()

        if self.webcam_active:
            self.root.after(30, self.capture_from_webcam)

    def show_image(self):
        """Отображение изображения на холсте."""
        if self.image is None:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        height, width = self.image.shape[:2]
        ratio = min(canvas_width / width, canvas_height / height)
        new_width, new_height = int(width * ratio), int(height * ratio)

        resized_image = cv2.resize(self.image, (new_width, new_height))
        img_pil = Image.fromarray(resized_image)
        self.tk_image = ImageTk.PhotoImage(img_pil)

        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            anchor=tk.CENTER,
            image=self.tk_image
        )

    def update_channels(self):
        """Обновление отображаемых цветовых каналов с использованием PyTorch."""
        if self.original_image is None:
            return

        channel = self.channel_var.get()
        image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Конвертируем в PyTorch tensor
        tensor_image = torch.from_numpy(image_rgb).float() / 255.0
        
        if channel == "RGB":
            self.image = image_rgb
        else:
            # Разделяем каналы с помощью PyTorch
            red, green, blue = tensor_image.unbind(dim=-1)
            zeros = torch.zeros_like(red)

            if channel == "R":
                result = torch.stack([red, zeros, zeros], dim=-1)
            elif channel == "G":
                result = torch.stack([zeros, green, zeros], dim=-1)
            elif channel == "B":
                result = torch.stack([zeros, zeros, blue], dim=-1)
            
            # Конвертируем обратно в numpy array
            self.image = (result.numpy() * 255).astype(np.uint8)

        self.show_image()

    def apply_negative(self):
        """Применение негатива к изображению с использованием PyTorch."""
        if self.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return

        # Конвертируем в PyTorch tensor
        tensor_image = torch.from_numpy(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)).float() / 255.0
        # Применяем негатив
        negative = 1.0 - tensor_image
        # Конвертируем обратно в numpy array
        self.image = (negative.numpy() * 255).astype(np.uint8)
        self.show_image()

    def rotate_image(self):
        """Поворот изображения на заданный угол."""
        if self.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return

        try:
            angle = float(self.rotate_entry.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректное число для угла")
            return

        height, width = self.original_image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB),
            rotation_matrix,
            (width, height)
        )

        self.image = rotated
        self.show_image()

    def draw_circle(self):
        """Рисование круга на изображении."""
        if self.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return

        try:
            x_coord = int(self.circle_x_entry.get())
            y_coord = int(self.circle_y_entry.get())
            radius = int(self.circle_radius_entry.get())

            self.image = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_BGR2RGB)
            cv2.circle(self.image, (x_coord, y_coord), radius, (255, 0, 0), 2)

            self.show_image()
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректный ввод: введите целые числа для координат и радиуса")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")

    def reset_image(self):
        """Сброс изображения к оригиналу."""
        if self.original_image is not None:
            self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.channel_var.set("RGB")
            self.show_image()

    def on_closing(self):
        """Обработчик закрытия окна."""
        self.stop_webcam()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()