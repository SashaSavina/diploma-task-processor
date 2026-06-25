import pandas as pd
import numpy as np
import joblib
import json
import re
import os
from datetime import datetime
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.corpus import stopwords
import pymorphy3
import nltk
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords', quiet=True)

class TaskProcessor:
    def __init__(self, models_path='./models'):
        self.models_path = models_path
        
        # пути к моделям
        self.classifier_path = os.path.join(models_path, 'classifier')
        self.regressor_path = os.path.join(models_path, 'regressor')
        self.improver_path = os.path.join(models_path, 'improver')
        
        # загрузка модели классификации
        print("Загрузка модели классификации...")
        self.classifier = joblib.load(os.path.join(self.classifier_path, 'xgb_model.pkl'))
        with open(os.path.join(self.classifier_path, 'feature_columns.json'), 'r', encoding='utf-8') as f:
            self.classifier_features = json.load(f)
        with open(os.path.join(self.classifier_path, 'target_mapping.json'), 'r', encoding='utf-8') as f:
            self.target_mapping = json.load(f)
        self.tfidf_clf = joblib.load(os.path.join(self.classifier_path, 'tfidf_vectorizer_1000.pkl'))
        
        # загрузка модели регрессии
        print("Загрузка модели регрессии...")
        self.regressor = joblib.load(os.path.join(self.regressor_path, 'lightgbm_regression_model.pkl'))
        
        # загрузка полного списка признаков регрессии
        with open(os.path.join(self.regressor_path, 'feature_columns.json'), 'r', encoding='utf-8') as f:
            self.feature_columns_reg = json.load(f)
        
        print(f"Загружено {len(self.feature_columns_reg)} признаков для регрессии")
        
        self.tfidf_reg = joblib.load(os.path.join(self.regressor_path, 'tfidf_vectorizer_regression.pkl'))
        self.tfidf_comment_reg = joblib.load(os.path.join(self.regressor_path, 'tfidf_comment_vectorizer_regression.pkl'))

        # загрузка модели улучшения формулировок
        print("Загрузка модели улучшения формулировок...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if os.path.exists(self.improver_path) and os.path.exists(os.path.join(self.improver_path, 'config.json')):
            print(f"Загрузка дообученной модели из {self.improver_path}")
            self.tokenizer = T5Tokenizer.from_pretrained(self.improver_path, legacy=False)
            self.improver = T5ForConditionalGeneration.from_pretrained(self.improver_path)
        else:
            print(f"Модель не найдена в {self.improver_path}")
            raise Exception("Модель улучшения не найдена")

        self.improver = self.improver.to(self.device)
        self.improver.eval()
        self.improver_prefix = "структурируй задачу: "

        # тест модели
        with torch.no_grad():
            test_input = f"{self.improver_prefix}тест"
            test_tokens = self.tokenizer(test_input, return_tensors="pt", max_length=64, truncation=True)
            test_tokens = {k: v.to(self.device) for k, v in test_tokens.items()}
            test_output = self.improver.generate(**test_tokens, max_new_tokens=30, num_beams=2)
            test_text = self.tokenizer.decode(test_output[0], skip_special_tokens=True)
            print(f"Тест модели улучшения: '{test_text}'")
        
        # инициализация для предобработки текста
        self.russian_stopwords = set(stopwords.words('russian'))
        self.morph = pymorphy3.MorphAnalyzer()
        
        # маппинг приоритетов
        self.priority_map = {
            'Бэклог задач': 0, 'Null': 0, 'Обычная': 1,
            'Тестирование': 2, 'Текущая итерация': 3,
            'Критическая': 4, 'Авария': 5
        }
        
        self.default_author = 'Руководитель проекта'
        self.default_priority = 'Обычная'
        
        print("Все модели загружены!\n")
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ''
        text = text.lower()
        text = re.sub(r'http[s]?://\S+', ' ссылка ', text)
        text = re.sub(r'\S+@\S+', ' емайл ', text)
        text = re.sub(r'@\w+', ' упоминание ', text)
        text = re.sub(r'\d{1,2}\.\d{1,2}\.\d{2,4}', ' дата ', text)
        text = re.sub(r'\[\d{2}\.\d{2}\.\d{4}\s\d{2}:\d{2}\]', ' датавремя ', text)
        text = re.sub(r'\d{1,2}:\d{2}', ' время ', text)
        text = re.sub(r'\d{1,3}[-]\d{1,3}[-]\d{2,4}', ' телефон ', text)
        text = re.sub(r'!?\[.*?]\(.*?\)', ' изображение ', text)
        text = re.sub(r'[^а-яА-Яa-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def clean_text_simple(self, text):
        if not isinstance(text, str):
            return ''
        text = text.lower()
        text = re.sub(r'http[s]?://\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        text = re.sub(r'@\w+', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'[^а-яА-Яa-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def lemmatize_text(self, text):
        if not isinstance(text, str) or text == '':
            return ''
        words = text.split()
        lemmatized = []
        for word in words:
            if word in self.russian_stopwords:
                continue
            if word in ['ссылка', 'емейл', 'упоминание', 'дата', 'время', 
                        'телефон', 'изображение', 'датавремя']:
                lemmatized.append(word)
                continue
            p = self.morph.parse(word)[0]
            lemmatized.append(p.normal_form)
        return ' '.join(lemmatized)
    
    def extract_text_features(self, text):
        features = {}
        features['len_text'] = len(str(text))
        features['word_count'] = len(str(text).split())
        features['has_digits'] = 1 if any(c.isdigit() for c in str(text)) else 0
        features['digit_count'] = sum(c.isdigit() for c in str(text))
        features['has_url'] = 1 if 'http' in str(text) or 'www' in str(text) else 0
        features['has_email'] = 1 if '@' in str(text) and '.' in str(text) else 0
        features['has_mention'] = 1 if '@' in str(text) else 0
        caps = sum(1 for c in str(text) if c.isupper())
        features['caps_ratio'] = caps / (len(str(text)) + 1)
        return features
    
    def extract_date_features(self, date_str=None):
        if date_str:
            try:
                for fmt in ['%Y-%m-%d', '%d.%m.%Y', '%Y/%m/%d', '%d/%m/%Y', 
                            '%Y-%m-%d %H:%M:%S', '%d.%m.%Y %H:%M:%S']:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    dt = datetime.now()
            except Exception:
                dt = datetime.now()
        else:
            dt = datetime.now()
        
        features = {
            'day_of_week': dt.weekday(),
            'month': dt.month,
            'hour': dt.hour,
            'is_weekend': 1 if dt.weekday() >= 5 else 0,
            'is_month_end': 1 if dt.day >= 25 else 0
        }
        return features, dt
    
    def _count_comments(self, comments):
        """Подсчитывает количество комментариев по тексту"""
        if not comments or not isinstance(comments, str):
            return 0
        
        timestamps = re.findall(r'\(\d{10,13}\)', comments)
        if timestamps:
            return len(timestamps)
        
        lines = comments.strip().split('\n')
        count = len([line for line in lines if line.strip()])
        
        return count if count > 0 else 0
    
    def improve_title(self, title):
        if not title or title.strip() == "":
            return title
        
        input_text = f"{self.improver_prefix}{title}"
        print(f"DEBUG improve_title: вход = '{input_text[:100]}'")
        
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=128, 
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.improver.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=64,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                temperature=0.7,
                do_sample=False
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"DEBUG improve_title: результат = '{result}'")
        
        if result.startswith(self.improver_prefix):
            result = result.replace(self.improver_prefix, "").strip()
        
        if not result or len(result) < 5:
            print("DEBUG improve_title: результат некорректен, возвращаем исходный")
            return title
        
        return result
    
    def classify_task(self, title, description, comments, author, priority):
        full_text = f"{title} {description} {comments}"
        full_text_clean = self.clean_text(full_text)
        full_text_lemmatized = self.lemmatize_text(full_text_clean)
        
        print(f"DEBUG classify_task: длина текста = {len(full_text_lemmatized)}")
        
        if not full_text_lemmatized.strip():
            full_text_lemmatized = "пустой текст"
        
        try:
            tfidf_vector = self.tfidf_clf.transform([full_text_lemmatized])
        except Exception as e:
            print(f"DEBUG classify_task: ошибка transform: {e}")
            raise
        
        tfidf_features = {}
        for i, word in enumerate(self.tfidf_clf.get_feature_names_out()):
            tfidf_features[f'tfidf_{word}'] = tfidf_vector[0, i]
        
        text_features = self.extract_text_features(full_text)
        
        executor_cols = []
        author_cols = []
        for col in self.classifier_features:
            if col.startswith('executor_'):
                executor_cols.append(col)
            elif col.startswith('author_'):
                author_cols.append(col)
        
        executor_features = {col: 0 for col in executor_cols}
        author_features = {col: 0 for col in author_cols}
        author_col_name = f"author_{author}"
        if author_col_name in author_features:
            author_features[author_col_name] = 1
        
        priority_num = self.priority_map.get(priority, 1)
        priority_feature = {'priority_num': priority_num}
        numeric_features = {'comments_count': 0}
        
        all_features = {}
        all_features.update(text_features)
        all_features.update(tfidf_features)
        all_features.update(executor_features)
        all_features.update(author_features)
        all_features.update(numeric_features)
        all_features.update(priority_feature)
        
        X_input = pd.DataFrame([all_features])
        for col in self.classifier_features:
            if col not in X_input.columns:
                X_input[col] = 0
        X_input = X_input[self.classifier_features]
        
        pred_encoded = self.classifier.predict(X_input)[0]
        category = self.target_mapping.get(str(pred_encoded), "Неизвестно")
        print(f"DEBUG classify_task: предсказанная категория = {category}")
        return category
    
    def predict_hours(self, title, description, comments, author, priority, category, comments_count=0, created_date=None):
        date_features, dt = self.extract_date_features(created_date)
        
        full_text = f"{title} {description}"
        
        text_for_tfidf = self.lemmatize_text(self.clean_text(full_text))
        comments_for_tfidf = self.clean_text_simple(comments)
        
        tfidf_vector = self.tfidf_reg.transform([text_for_tfidf])
        tfidf_features = {}
        for i, word in enumerate(self.tfidf_reg.get_feature_names_out()):
            tfidf_features[f'tfidf_{word}'] = tfidf_vector[0, i]
        
        tfidf_comment_vector = self.tfidf_comment_reg.transform([comments_for_tfidf])
        tfidf_comment_features = {}
        for i, word in enumerate(self.tfidf_comment_reg.get_feature_names_out()):
            tfidf_comment_features[f'tfidf_comm_{word}'] = tfidf_comment_vector[0, i]
        
        text_features = self.extract_text_features(full_text)
        text_features['len_title'] = len(str(title))
        
        all_features = {}
        all_features.update(text_features)
        all_features.update(tfidf_features)
        all_features.update(tfidf_comment_features)
        
        for col in self.feature_columns_reg:
            all_features[col] = 0
        
        author_col_name = f"author_{author}"
        if author_col_name in all_features:
            all_features[author_col_name] = 1
        
        task_type_col_name = f"type_{category}"
        if task_type_col_name in all_features:
            all_features[task_type_col_name] = 1
        
        all_features['priority_num'] = self.priority_map.get(priority, 1)
        all_features['comments_count'] = comments_count
        all_features['timeline_marks'] = 0
        all_features['deviation'] = 0
        
        all_features.update(date_features)
        
        X_input = pd.DataFrame([all_features])
        X_input = X_input[self.feature_columns_reg]
        
        if X_input.shape[1] != 762:
            print(f"Предупреждение: создано {X_input.shape[1]} признаков, ожидается 762")
        
        hours = self.regressor.predict(X_input)[0]
        hours = round(hours * 2) / 2
        return hours, dt
    
    def process_task(self, task_data):
        title = task_data.get('наименование_задачи', '')
        description = task_data.get('описание_задачи', '')
        comments = task_data.get('комментарии', '')
        author = task_data.get('постановщик', self.default_author)
        priority = task_data.get('приоритет', self.default_priority)
        comments_count = self._count_comments(comments)
        created_date = task_data.get('дата_постановки', None)
        
        if not title or not description:
            raise ValueError("Необходимо указать наименование и описание задачи")
        
        improved_title = self.improve_title(title)
        print(f"DEBUG process_task: improved_title = '{improved_title}'")
        category = self.classify_task(improved_title, description, comments, author, priority)
        hours, dt = self.predict_hours(improved_title, description, comments, author, priority, category, comments_count, created_date)
        
        return {
            'улучшенное_наименование': improved_title,
            'исходное_описание': description,
            'категория_задачи': category,
            'оценка_в_часах': hours,
            'постановщик': author,
            'приоритет': priority,
            'дата_обработки': dt.strftime('%Y-%m-%d %H:%M:%S')
        }