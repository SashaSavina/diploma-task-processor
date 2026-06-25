import pandas as pd
import json
import os
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FileProcessor:
    def __init__(self, task_processor):
        self.processor = task_processor
        
        # словарь для поиска похожих названий колонок
        self.column_mapping = {
            'наименование_задачи': ['наименование', 'название', 'title', 'заголовок', 'name', 'task_name'],
            'описание_задачи': ['описание', 'description', 'текст', 'desc', 'detail'],
            'постановщик': ['постановщик', 'author', 'кто поставил', 'создатель', 'creator'],
            'приоритет': ['приоритет', 'priority', 'важность', 'importance', 'urgent'],
            'комментарии': ['комментарии', 'comments', 'обсуждение', 'comment', 'notes'],
            'дата_постановки': ['дата постановки', 'дата_создания', 'created_date', 'date', 'created_at']
        }
        
        self.default_values = {
            'постановщик': self.processor.default_author,
            'приоритет': self.processor.default_priority,
            'комментарии': '',
            'дата_постановки': None
        }
    
    def _count_comments(self, comments):
        """Подсчитывает количество комментариев по тексту"""
        if not comments or not isinstance(comments, str):
            return 0
        
        # ищем метки времени в формате (число)
        timestamps = re.findall(r'\(\d{10,13}\)', comments)
        if timestamps:
            return len(timestamps)
        
        # считаем непустые строки
        lines = comments.strip().split('\n')
        count = len([line for line in lines if line.strip()])
        
        return count if count > 0 else 0
    
    def _map_columns(self, columns):
        """Определяет, какой колонке какой тип данных соответствует"""
        mapped = {}
        for target, possible_names in self.column_mapping.items():
            for col in columns:
                col_lower = col.lower().strip()
                if any(name.lower() in col_lower or col_lower == name.lower() for name in possible_names):
                    mapped[target] = col
                    break
        return mapped
    
    def _normalize_row(self, row, mapped_columns):
        normalized = {}
        
        # заполняем значения по умолчанию
        for field in self.column_mapping.keys():
            normalized[field] = self.default_values.get(field)
        
        # перезаписываем найденные поля
        for field, col_name in mapped_columns.items():
            if col_name in row.index:
                value = row[col_name]
                
                # обработка пустых значений
                if pd.isna(value):
                    normalized[field] = self.default_values.get(field)
                    continue
                
                # поле описание оставляем как есть (не разбиваем)
                if field == 'описание_задачи':
                    normalized[field] = str(value)
                elif field == 'дата_постановки':
                    normalized[field] = value if value else None
                else:
                    normalized[field] = str(value).strip() if value else self.default_values.get(field)
        
        # проверяем обязательные поля
        if 'наименование_задачи' not in mapped_columns:
            raise ValueError("В файле не найдена колонка с наименованием задачи")
        if 'описание_задачи' not in mapped_columns:
            raise ValueError("В файле не найдена колонка с описанием задачи")
        
        # количество комментариев вычисляется автоматически из текста
        comments = normalized.get('комментарии', '')
        normalized['количество_комментариев'] = self._count_comments(comments)
        
        return normalized
    
    def process_csv(self, file_path):
        """Обработка CSV файла с автоопределением разделителя"""
        # пробуем прочитать с разными разделителями
        df = None
        for delimiter in [',', ';', '\t']:
            try:
                test_df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8', nrows=1)
                if len(test_df.columns) > 1:
                    df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
                    print(f"CSV прочитан с разделителем: '{delimiter}'")
                    break
            except:
                continue
        
        if df is None:
            df = pd.read_csv(file_path, encoding='utf-8')
            print("CSV прочитан с автоопределением разделителя")
        
        print(f"Загружено {len(df)} строк из CSV")
        print(f"Колонки: {df.columns.tolist()}")
        return self._process_dataframe(df)
    
    def process_json(self, file_path):
        """Обработка JSON файла"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'tasks' in data:
            df = pd.DataFrame(data['tasks'])
        else:
            df = pd.DataFrame([data])
        
        print(f"Загружено {len(df)} строк из JSON")
        return self._process_dataframe(df)
    
    def _process_dataframe(self, df):
        """Обработка DataFrame (общая логика для CSV и JSON)"""
        mapped_columns = self._map_columns(df.columns)
        
        print(f"\nОпределены колонки:")
        for field, col in mapped_columns.items():
            print(f"  {field} -> {col}")
        
        results = []
        for idx, row in df.iterrows():
            try:
                task_data = self._normalize_row(row, mapped_columns)
                
                title = str(task_data.get('наименование_задачи', ''))[:60]
                print(f"\nОбработка задачи {idx + 1}/{len(df)}: {title}...")
                
                result = self.processor.process_task(task_data)
                
                # добавляем исходные данные для справки
                result['исходное_наименование'] = task_data.get('наименование_задачи', '')
                results.append(result)
                print(f"  Категория: {result['категория_задачи']}")
                print(f"  Оценка: {result['оценка_в_часах']} часов")
                print(f"  Комментариев: {task_data.get('количество_комментариев', 0)}")
                
            except Exception as e:
                error_msg = str(e)
                print(f"  ОШИБКА: {error_msg}")
                # получаем наименование задачи для отчёта
                task_name = ''
                if 'наименование_задачи' in mapped_columns:
                    task_name = str(row.get(mapped_columns['наименование_задачи'], ''))
                results.append({
                    'ошибка': error_msg,
                    'исходное_наименование': task_name,
                    'исходное_описание': str(row.get(mapped_columns.get('описание_задачи', ''), ''))
                })
        
        return results
    
    def save_results(self, results, output_path, input_format='json'):
        """Сохранение результатов в файл"""
        # создаём папку, если её нет
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # определяем формат по расширению, если не указан
        if input_format is None and output_path:
            if output_path.endswith('.csv'):
                input_format = 'csv'
            else:
                input_format = 'json'
        
        if input_format == 'csv':
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nРезультаты сохранены в {output_path}")