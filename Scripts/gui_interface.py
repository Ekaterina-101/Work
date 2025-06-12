import sys #для работы с системными путями
from pathlib import Path #для работы с файловыми системами
import os #для взаимодействия с операционной системой (например, создание папок)
# from datetime import datetime

# Получаем абсолютный путь к корню проекта
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import tkinter as tk #для создания графического интерфейса
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #для построения графиков
import matplotlib.pyplot as plt
import seaborn as sns #для визуализации данных

#собственная реализация функций
from Library.data_processing import (
    load_config, get_config, set_config_value,
    load_data, plot_survival, plot_family_size,
    plot_survival_by_pclass, plot_survival_by_age,
    train_model, predict_test_data, generate_pivot_report
)

root = None
main_frame = None
analysis_frame = None
model_frame = None
report_frame = None
figure = None
canvas = None
model_info = None
report_text = None
current_plot_name = None


#Создает лавное меню приложения
def create_menu():
    menubar = tk.Menu(root)
    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Главная", command=lambda: show_frame(main_frame))
    file_menu.add_separator()
    file_menu.add_command(label="Выход", command=root.quit)
    menubar.add_cascade(label="Файл", menu=file_menu)

    analysis_menu = tk.Menu(menubar, tearoff=0)
    analysis_menu.add_command(label="Анализ данных", command=lambda: show_frame(analysis_frame))
    menubar.add_cascade(label="Анализ", menu=analysis_menu)

    model_menu = tk.Menu(menubar, tearoff=0)
    model_menu.add_command(label="Модель", command=lambda: show_frame(model_frame))
    menubar.add_cascade(label="Модель", menu=model_menu)

    report_menu = tk.Menu(menubar, tearoff=0)
    report_menu.add_command(label="Отчеты", command=lambda: show_frame(report_frame))
    menubar.add_cascade(label="Отчеты", menu=report_menu)

    settings_menu = tk.Menu(menubar, tearoff=0)
    settings_menu.add_command(label="Настройки", command=open_settings)
    menubar.add_cascade(label="Настройки", menu=settings_menu)

    root.config(menu=menubar)


#Создание стартового экрана
def create_main_frame():
    global main_frame
    main_frame = ttk.Frame(root, padding="10")
    ttk.Label(main_frame, text="Анализ данных Titanic", font=('Arial', 40)).pack(pady=20)
    ttk.Label(main_frame, text="Добро пожаловать в приложение для анализа данных пассажиров Титаника",
              wraplength=800, font=('Arial', 20), justify='center').pack(pady=10)
    ttk.Label(main_frame, text="Функционал приложения:", font=('Arial', 20)).pack(pady=10, anchor='w')

    features = [
        "Загрузка и предварительный анализ данных",
        "Построение модели предсказания выживания пассажиров",
        "Генерация отчетов и сохранение результатов"
    ]
    for feature in features:
        ttk.Label(main_frame, text=f"• {feature}", font=('Arial', 18)).pack(anchor='w')

    ttk.Button(main_frame, text="Начать анализ",
               command=start_analysis, style='Large.TButton').pack(pady=20)


#Загружает данные и переходит на анализ
def start_analysis():
    status, message = load_data()
    if status == "Успех":
        show_frame(analysis_frame)
    else:
        messagebox.showerror("Ошибка", message)


#Сохраняет текущий график
def save_current_plot():
    global current_plot_name

    if not current_plot_name:
        messagebox.showerror("Ошибка", "Нет активного графика для сохранения")
        return

    try:
        config = get_config()
        default_graphics_dir = str(project_root / "graphics")
        graphics_dir = config['DEFAULT'].get('graphics_dir', '').strip()
        if not graphics_dir:
            graphics_dir = default_graphics_dir

        # Создаем папку, если ее нет
        os.makedirs(graphics_dir, exist_ok=True)

        filename = f"{graphics_dir}/{current_plot_name}.png"

        figure.savefig(filename, dpi=300, bbox_inches='tight')
        messagebox.showinfo("Успех", f"График сохранен в:\n{filename}")
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось сохранить график:\n{str(e)}")


#установка имени и рисовка графиков
def plot_and_set_name(plot_func, plot_name, *args):
    global current_plot_name
    current_plot_name = plot_name
    plot_func(*args)


#выбор графика
def create_analysis_frame():
    global analysis_frame, figure, canvas

    analysis_frame = ttk.Frame(root, padding="10")
    ttk.Label(analysis_frame, text="Анализ данных", font=('Arial', 20)).grid(row=0, column=0, columnspan=3, pady=10)

    ttk.Button(analysis_frame, text="Выживание по полу",
               command=lambda: plot_and_set_name(plot_survival, "survival_by_sex", 'Sex', figure, canvas),
               style='Medium.TButton'
               ).grid(row=1, column=0, pady=5, padx=5, sticky='ew')

    ttk.Button(analysis_frame, text="Размер семьи и выживание",
               command=lambda: plot_and_set_name(plot_family_size, "survival_by_family_size", figure, canvas),
               style='Medium.TButton'
               ).grid(row=1, column=1, pady=5, padx=5, sticky='ew')

    ttk.Button(analysis_frame, text="Выживание по классу кают",
               command=lambda: plot_and_set_name(plot_survival_by_pclass, "survival_by_pclass", figure, canvas),
               style='Medium.TButton'
               ).grid(row=2, column=0, pady=5, padx=5, sticky='ew')

    ttk.Button(analysis_frame, text="Выживание по возрасту",
               command=lambda: plot_and_set_name(plot_survival_by_age, "survival_by_age", figure, canvas),
               style='Medium.TButton'
               ).grid(row=2, column=1, pady=5, padx=5, sticky='ew')

    ttk.Button(analysis_frame, text="Сохранить график",
               command=save_current_plot,
               style='Medium.TButton'
               ).grid(row=3, column=0, columnspan=2, pady=10, sticky='nsew')


    figure = plt.figure(figsize=(8, 6), dpi=100)
    canvas = FigureCanvasTkAgg(figure, master=analysis_frame)
    canvas.get_tk_widget().grid(row=4, column=0, columnspan=3, pady=10)

    ttk.Button(analysis_frame, text="Назад",
               command=lambda: show_frame(main_frame), style='Medium.TButton').grid(row=5, column=0, pady=10, sticky='w')
    ttk.Button(analysis_frame, text="К модели",
               command=lambda: show_frame(model_frame), style='Medium.TButton').grid(row=5, column=2, pady=10, sticky='e')


#создание фрейма для работы с моделью
def create_model_frame():
    global model_frame, model_info
    model_frame = ttk.Frame(root, padding="10")
    ttk.Label(model_frame, text="Модель предсказания", font=('Arial', 20)).grid(row=0, column=0, columnspan=2, pady=10)

    model_info = tk.Text(model_frame, height=10, width=60, wrap=tk.WORD, font=('Arial', 14))
    model_info.grid(row=1, column=0, columnspan=2, pady=5)
    model_info.insert(tk.END, "Модель не обучена. Загрузите данные и обучите модель.")
    model_info.config(state=tk.DISABLED)

    ttk.Button(model_frame, text="Обучить модель",
               command=train_model_gui, style='Large.TButton').grid(row=2, column=0, pady=5, sticky='ew')
    ttk.Button(model_frame, text="Предсказать на тестовых данных",
               command=predict_test_data_gui, style='Large.TButton').grid(row=2, column=1, pady=5, sticky='ew')

    ttk.Button(model_frame, text="Назад",
               command=lambda: show_frame(analysis_frame), style='Medium.TButton').grid(row=3, column=0, pady=10, sticky='w')
    ttk.Button(model_frame, text="К отчетам",
               command=lambda: show_frame(report_frame), style='Medium.TButton').grid(row=3, column=1, pady=10, sticky='e')


#создание фрейма для отчетов
def create_report_frame():
    global report_frame, report_text
    report_frame = ttk.Frame(root, padding="10")
    ttk.Label(report_frame, text="Сводная таблица", font=('Arial', 20)).grid(row=0, column=0, columnspan=2, pady=10)

    ttk.Button(report_frame, text="Показать сводную таблицу",
               command=lambda: generate_pivot_report(report_text), style='Large.TButton').grid(row=1, column=0, pady=5, sticky='ew')

    report_text = tk.Text(report_frame, height=15, width=70, wrap=tk.WORD, font=('Arial', 14))
    report_text.grid(row=2, column=0, columnspan=2, pady=10)

    ttk.Button(report_frame, text="Назад",
               command=lambda: show_frame(model_frame), style='Medium.TButton').grid(row=3, column=0, pady=10, sticky='w')
    ttk.Button(report_frame, text="На главную",
               command=lambda: show_frame(main_frame), style='Medium.TButton').grid(row=3, column=1, pady=10, sticky='e')


#скрывает предыдушие фреймы
def hide_all_frames():
    for frame in [main_frame, analysis_frame, model_frame, report_frame]:
        if frame:
            frame.pack_forget()


#показывает текущий фрейм
def show_frame(frame):
    hide_all_frames()
    if frame:
        frame.pack(fill=tk.BOTH, expand=True)


#обучает модель
def train_model_gui():
    status, message = train_model()
    if status == "Успех":
        model_info.config(state=tk.NORMAL)
        model_info.delete(1.0, tk.END)
        model_info.insert(tk.END, message)
        model_info.config(state=tk.DISABLED)
        messagebox.showinfo("Успех", message)
    else:
        messagebox.showerror("Ошибка", message)


#предсказание на новых данных
def predict_test_data_gui():
    status, message = predict_test_data()
    if status == "Успех":
        messagebox.showinfo("Успех", message)
    else:
        messagebox.showerror("Ошибка", message)


#создает окно настроек
def open_settings():
    current_config = get_config()
    settings_window = tk.Toplevel(root)
    settings_window.title("Настройки")
    settings_window.geometry("450x400")

    ttk.Label(settings_window, text="Настройки приложения", font=('Arial', 16)).pack(pady=10)

    # Папка с данными
    ttk.Label(settings_window, text="Папка с данными:").pack(anchor='w', padx=20)
    data_dir_entry = ttk.Entry(settings_window, width=50)
    data_dir_entry.pack(padx=20, pady=5, fill=tk.X)
    data_dir_entry.insert(0, current_config['DEFAULT'].get('data_dir', ''))

    # Папка для отчетов
    ttk.Label(settings_window, text="Папка для отчетов:").pack(anchor='w', padx=20)
    output_dir_entry = ttk.Entry(settings_window, width=50)
    output_dir_entry.pack(padx=20, pady=5, fill=tk.X)
    output_dir_entry.insert(0, current_config['DEFAULT'].get('output_dir', ''))

    # Папка для графиков
    default_graphics_dir = str(project_root / "graphics")
    ttk.Label(settings_window, text="Папка для графиков:").pack(anchor='w', padx=20)
    graphics_dir_entry = ttk.Entry(settings_window, width=50)
    graphics_dir_entry.pack(padx=20, pady=5, fill=tk.X)
    graphics_dir_entry.insert(0, current_config['DEFAULT'].get('graphics_dir', default_graphics_dir))

    def save_settings():
        new_data_dir = data_dir_entry.get()
        new_output_dir = output_dir_entry.get()
        new_graphics_dir = graphics_dir_entry.get()
        set_config_value('DEFAULT', 'data_dir', new_data_dir)
        set_config_value('DEFAULT', 'output_dir', new_output_dir)
        set_config_value('DEFAULT', 'graphics_dir', new_graphics_dir)
        settings_window.destroy()
        messagebox.showinfo("Успех", "Настройки сохранены!")

    ttk.Button(settings_window, text="Сохранить", command=save_settings).pack(pady=10)
    ttk.Button(settings_window, text="Отмена", command=settings_window.destroy).pack(pady=5)


if __name__ == "__main__":
    root = tk.Tk() #создание главного окна

    style = ttk.Style() #размер кнопок
    style.configure('Medium.TButton', font=('Arial', 15))
    style.configure('Large.TButton', font=('Arial', 17), padding=10)

    load_config()
    create_menu()
    create_main_frame()
    create_analysis_frame()
    create_model_frame()
    create_report_frame()
    main_frame.pack(fill=tk.BOTH, expand=True)
    root.title("Анализ данных Titanic")
    root.geometry("820x1000")
    root.mainloop()
