import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import configparser
import os

#глобальные переменные
config = None
config_file = 'config.ini'
train_df = None
test_df = None
model = None


#загрузка конфигурационного файла
def load_config():
    global config
    config = configparser.ConfigParser()
    if os.path.exists(config_file):
        config.read(config_file)
    else:
        config['DEFAULT'] = {
            'data_dir': './Data',
            'output_dir': 'output',
            'font': 'Arial',
            'font_size': '10',
            'theme': 'light'
        }
        save_config()


#сохранение конфигурационного файла
def save_config():
    global config
    with open(config_file, 'w') as f:
        config.write(f)


#вспомогательная функция
def get_config():
    return config


#вспомогательная функция
def set_config_value(section, key, value):
    global config
    if section not in config:
        config[section] = {}
    config[section][key] = value
    save_config()


#считывание данных
def load_data():
    global train_df, test_df
    try:
        data_dir = config['DEFAULT'].get('data_dir', './Data')
        train_path = os.path.join(data_dir, 'train.csv')
        test_path = os.path.join(data_dir, 'test.csv')

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        for df in [train_df, test_df]:
            df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
            df['FamilySize'] = df['SibSp'] + df['Parch']

        return "Успех", "Данные успешно загружены и обработаны!"
    except Exception as e:
        return "Ошибка", f"Не удалось загрузить данные: {str(e)}"


#график
def plot_survival(column, figure, canvas):
    global train_df
    if train_df is None:
        return "Ошибка", "Сначала загрузите данные!"

    COLUMN_NAMES_RU = {
        'Sex': 'Пол',
        'Pclass': 'Класс',
        'Age': 'Возраст',
        'FamilySize': 'Размер семьи'
    }

    figure.clear()
    ax = figure.add_subplot(111)

    survival_palette = {0: 'pink', 1: 'palegreen'}  # розовый - нет, фисташковый - да

    if column == 'Sex':
        temp_df = train_df.copy()
        temp_df['Sex'] = temp_df['Sex'].map({'male': 'мужской', 'female': 'женский'})
        sns.countplot(data=temp_df, x=column, hue='Survived', ax=ax, palette=survival_palette)
    else:
        sns.countplot(data=train_df, x=column, hue='Survived', ax=ax, palette=survival_palette)

    ax.set_title(f'Выживание по {COLUMN_NAMES_RU.get(column, column)}')
    ax.set_xlabel(COLUMN_NAMES_RU.get(column, column))
    ax.set_ylabel('Число пассажиров')
    ax.legend(title='Выжил', labels=['Нет', 'Да'])
    canvas.draw()
    return "Успех", ""


#график выживаемости в зависимости от размера семьи
def plot_family_size(figure, canvas):
    global train_df
    if train_df is None:
        return "Ошибка", "Сначала загрузите данные!"

    figure.clear()
    ax = figure.add_subplot(111)

    sns.countplot(data=train_df, x='FamilySize', hue='Survived', ax=ax, palette={0: 'pink', 1: 'palegreen'})

    ax.set_title('Выживание в зависимости от размера семьи')
    ax.set_xlabel('Количество родственников')
    ax.set_ylabel('Число пассажиров')
    ax.legend(title='Выжил', labels=['Нет', 'Да'])
    canvas.draw()
    return "Успех", ""


#график выживаемости по классу каюты
def plot_survival_by_pclass(figure, canvas):
    global train_df
    if train_df is None:
        return "Ошибка", "Сначала загрузите данные!"

    figure.clear()
    ax = figure.add_subplot(111)

    sns.countplot(data=train_df, x='Pclass', hue='Survived', ax=ax, palette={0: 'pink', 1: 'palegreen'})

    ax.set_title('Выживание по классу каюты')
    ax.set_xlabel('Класс каюты')
    ax.set_ylabel('Число пассажиров')
    ax.legend(title='Выжил', labels=['Нет', 'Да'])
    canvas.draw()
    return "Успех", ""


#график выживаемости по возрасу
def plot_survival_by_age(figure, canvas):
    global train_df
    if train_df is None:
        return "Ошибка", "Сначала загрузите данные!"

    figure.clear()

    ax1 = figure.add_subplot(121)  # левый график (мужчины)
    ax2 = figure.add_subplot(122)  # правый график (женщины)

    males = train_df[train_df['Sex'] == 'male']
    females = train_df[train_df['Sex'] == 'female']

    survival_palette = {0: 'pink', 1: 'palegreen'}  # розовый - нет, фисташковый - да

    #общие настройки для обоих графиков
    plot_params = {
        'bins': range(0, 81, 5),  # возраст от 0 до 80 с шагом 5
        'kde': False,
        'palette': survival_palette,
        'alpha': 0.7,
        'edgecolor': 'black',
        'linewidth': 0.5
    }

    #график для мужчин
    sns.histplot(
        data=males,
        x='Age',
        hue='Survived',
        **plot_params,
        ax=ax1
    )
    ax1.set_title('Выживание мужчин по возрасту', pad=20)
    ax1.set_xlabel('Возраст')
    ax1.set_ylabel('Число пассажиров', labelpad=10)
    ax1.legend(title='Выжил', labels=['Нет', 'Да'], framealpha=0.7)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    #график для женщин
    sns.histplot(
        data=females,
        x='Age',
        hue='Survived',
        **plot_params,
        ax=ax2
    )
    ax2.set_title('Выживание женщин по возрасту', pad=20)
    ax2.set_xlabel('Возраст')
    ax2.set_ylabel('Число пассажиров', labelpad=10)
    ax2.legend(title='Выжил', labels=['Нет', 'Да'], framealpha=0.7)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    #настройка отступов и размера
    figure.tight_layout(pad=3.0)
    figure.subplots_adjust(top=0.85)

    canvas.draw()
    return "Успех", ""


#функция для обучения модели
def train_model():
    global train_df, model
    if train_df is None:
        return "Ошибка", "Сначала загрузите данные!"

    try:
        #разделяем признаки Х и целевую переменную У
        X = train_df.drop('Survived', axis=1) #все столбцы кроме Survived
        y = train_df['Survived'] #целевая переменная

        #категориальные и числовые признаки
        categorical_cols = ['Sex', 'Embarked']
        numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']

        #применим преобразования к признакам
        preprocessor = ColumnTransformer(
            transformers=[
                #для числовых
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')), #заполняем медианой
                    ('scaler', StandardScaler()) #нормализуем значения среднее = 0, дисперсия = 1
                ]), numerical_cols),
                #для категориальных
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')), #заполняем самым частым значением
                    ('onehot', OneHotEncoder(handle_unknown='ignore')) #применяем OneHotEncoding, делает из категориальных числовые
                ]), categorical_cols)
            ])

        #создаем конвеер Pipeline, сначала предобработка, затем обучение модели
        model = Pipeline([
            ('preprocessor', preprocessor), #предобработка данных
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)) #обучаем RandomForest, фиксируем случайность для воспроизводимости
        ])

        #делим данные на обучающую 80% и валидационную 20% выборки
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train) #обучаем модель на обучающей выборке
        y_pred = model.predict(X_val) #делаем предсказания на валидационной выборке
        accuracy = accuracy_score(y_val, y_pred) #вычисляем точность модели - долю правильных предсказаний

        #текстовый результат для отображения
        result = (
            f"Модель RandomForestClassifier обучена.\n"
            f"Точность на валидации: {accuracy:.4f}\n"
            f"Параметры модели:\n"
            f"Количество деревьев: 100\n"
            f"Случайное состояние: 42"
        )
        return "Успех", result
    except Exception as e:
        return "Ошибка", f"Не удалось обучить модель: {str(e)}"


#функция для предсказания
def predict_test_data():
    global model, test_df
    #проверяем, обучена ли модель
    if model is None:
        return "Ошибка", "Сначала обучите модель!"
    #проверяем, загружены ли тестовые данные
    if test_df is None:
        return "Ошибка", "Сначала загрузите тестовые данные!"

    try:
        test_ids = test_df['PassengerId'] #получаем идентификаторы пассажиров из тестового набора
        X_test = test_df
        preds = model.predict(X_test) #делаем предсказания
        output_dir = config['DEFAULT'].get('output_dir', 'output')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'submission.csv')
        output = pd.DataFrame({'PassengerId': test_ids, 'Survived': preds}) #сохраняем резульататы в csv
        output.to_csv(output_path, index=False)
        return "Успех", f"Результаты сохранены в файл: {output_path}"
    except Exception as e:
        return "Ошибка", f"Не удалось выполнить предсказание: {str(e)}"


def generate_pivot_report(report_text_widget):
    global train_df
    #проверяем, загружены ли данные
    if train_df is None:
        return "Ошибка", "Сначала загрузите данные!"

    try:
        #вероятность выживания по классу и полу
        pivot = pd.pivot_table(train_df, values='Survived', index='Pclass', columns='Sex', aggfunc='mean')
        report_text_widget.config(state='normal')
        report_text_widget.delete(1.0, 'end')
        report_text_widget.insert('end', "Сводная таблица (вероятность выживания по классу и полу):\n")
        report_text_widget.insert('end', pivot.to_string())
        report_text_widget.config(state='disabled')
        return "Успех", ""
    except Exception as e:
        return "Ошибка", f"Не удалось сгенерировать отчет: {str(e)}"