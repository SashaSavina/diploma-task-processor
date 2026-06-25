import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime

st.set_page_config(
    page_title="Task Processing Service",
    page_icon="",
    layout="wide"
)

# Скрываем меню с тремя точками и deploy кнопку
st.markdown("""
<style>
    /* Скрываем меню с тремя точками */
    #MainMenu {visibility: hidden;}
    /* Скрываем deploy кнопку */
    .stDeployButton {display: none;}
    /* Скрываем footer */
    footer {visibility: hidden;}
    /* Скрываем баннер "Made with Streamlit" */
    .stApp > footer {display: none;}
    
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: #1a1a1a;
        border-radius: 0px;
        margin-bottom: 2rem;
        border-bottom: 2px solid #4a4a4a;
    }
    .main-header h1 {
        color: #ffffff;
        margin: 0;
        font-size: 1.75rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    .main-header p {
        color: #a0a0a0;
        font-size: 0.9rem;
        font-weight: 300;
        margin-top: 0.5rem;
    }
    .result-card {
        background: #f5f5f5;
        border-radius: 4px;
        padding: 1.5rem;
        margin-top: 1rem;
        border-left: 3px solid #4a4a4a;
        font-family: 'Courier New', monospace;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 4px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-card p {
        margin: 0;
        color: #666666;
        font-size: 0.85rem;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card h3 {
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        color: #1a1a1a;
        font-weight: 500;
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: normal;
        line-height: 1.3;
    }
    .stButton > button {
        background: #2c2c2c;
        color: white;
        border: none;
        border-radius: 2px;
        padding: 0.5rem 1rem;
        font-weight: 400;
        font-size: 0.9rem;
    }
    .stButton > button:hover {
        background: #1a1a1a;
        color: white;
        border: 1px solid #4a4a4a;
    }
    .stTextArea > div > textarea {
        border-radius: 2px;
        border: 1px solid #d0d0d0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
    .stSelectbox > div > div {
        border-radius: 2px;
    }
    .stDateInput > div > input {
        border-radius: 2px;
        font-family: 'Courier New', monospace;
    }
    .success-message {
        background: #e8e8e8;
        color: #1a1a1a;
        padding: 0.75rem;
        border-radius: 2px;
        margin: 1rem 0;
        border-left: 3px solid #4a4a4a;
        font-family: 'Courier New', monospace;
    }
    .error-message {
        background: #f0f0f0;
        color: #cc0000;
        padding: 0.75rem;
        border-radius: 2px;
        margin: 1rem 0;
        border-left: 3px solid #cc0000;
        font-family: 'Courier New', monospace;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f0f0f0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0px;
        padding: 0.5rem 1rem;
        font-weight: 400;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #1a1a1a;
    }
    .stAlert {
        border-radius: 2px;
        font-family: 'Courier New', monospace;
    }
    div[data-testid="stDataFrame"] {
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
    }
    .stMarkdown {
        font-family: 'Courier New', monospace;
    }
    label {
        font-weight: 400;
        color: #333333;
    }
    
    /* Улучшенный стиль для таблицы */
    .dataframe {
        font-size: 0.8rem;
        border-collapse: collapse;
        width: 100%;
    }
    .dataframe th {
        background-color: #1a1a1a;
        color: white;
        padding: 8px;
        text-align: left;
    }
    .dataframe td {
        padding: 6px;
        border-bottom: 1px solid #e0e0e0;
    }
    
    /* Русский текст для file uploader */
    .stFileUploader > div > div > button {
        background-color: #2c2c2c;
        color: white;
    }
</style>

""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>Интеллектуальный обработчик задач</h1>
    <p>Классификация | Оценка Трудозатрат | Улучшение Формулировок</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### О сервисе")
    st.markdown("""
    Используемые модели:
    
    - Улучшение формулировки
    - Классификация задачи
    - Оценка трудозатрат
    
    Поддерживаемые форматы: CSV, JSON
    """)

# Инициализация состояния для хранения последнего результата
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'last_improved_title' not in st.session_state:
    st.session_state.last_improved_title = None

tab1, tab2 = st.tabs(["Ручной ввод", "Загрузка файла"])

with tab1:
    st.markdown("### Введите данные задачи")
    
    col1, col2 = st.columns(2)
    
    with col1:
        title = st.text_area("Наименование задачи *", height=80)
        description = st.text_area("Описание задачи *", height=150)
    
    with col2:
        author = st.selectbox("Постановщик", [
            "Бекэнд разработчик",
            "Верстальщик",
            "Дизайнер",
            "Руководитель проекта",
            "Тестировщик",
            "Фронтенд разработчик"
        ])
        priority = st.selectbox("Приоритет", [
            "Обычная",
            "Критическая",
            "Авария",
            "Тестирование",
            "Текущая итерация",
            "Бэклог задач"
        ])
        comments = st.text_area("Комментарии", height=100)
        created_date = st.date_input("Дата постановки", value=datetime.now())
    
    if st.button("Обработать задачу", use_container_width=True):
        if not title or not description:
            st.markdown('<div class="error-message">Необходимо заполнить наименование и описание задачи</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Обработка задачи..."):
                try:
                    response = requests.post(
                        "http://localhost:8000/task",
                        json={
                            "наименование_задачи": title,
                            "описание_задачи": description,
                            "постановщик": author,
                            "приоритет": priority,
                            "комментарии": comments,
                            "дата_постановки": created_date.strftime("%Y-%m-%d")
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Делим оценку трудозатрат на 5
                        original_hours = result['оценка_в_часах']
                        adjusted_hours = round(original_hours / 5, 1)
                        result['оценка_в_часах'] = adjusted_hours
                        
                        st.session_state.last_result = result
                        st.session_state.last_improved_title = result['улучшенное_наименование']
                        
                        st.markdown('<div class="success-message">Задача обработана</div>', unsafe_allow_html=True)
                        
                        # Приводим к верхнему регистру первую букву
                        category_text = result['категория_задачи'].capitalize()
                        author_text = result['постановщик'].capitalize()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f'<div class="metric-card"><p>категория</p><h3>{category_text}</h3></div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown(f'<div class="metric-card"><p>трудозатраты</p><h3>{result["оценка_в_часах"]} ч</h3></div>', unsafe_allow_html=True)
                        with col3:
                            st.markdown(f'<div class="metric-card"><p>постановщик</p><h3>{author_text}</h3></div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        
                        # Улучшенное наименование с большой буквы
                        improved = result['улучшенное_наименование']
                        if improved:
                            improved = improved[0].upper() + improved[1:] if len(improved) > 1 else improved.upper()
                        st.markdown(f"**Улучшенное наименование:**")
                        st.markdown(f"{improved}")
                        st.markdown(f"**Дата обработки:** {result['дата_обработки']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        error_detail = response.json().get('detail', 'Неизвестная ошибка')
                        st.markdown(f'<div class="error-message">Ошибка: {error_detail}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="error-message">Не удалось подключиться к api: {e}</div>', unsafe_allow_html=True)
    
    # Кнопка скачивания результата ручного ввода
    if st.session_state.last_result is not None:
        st.markdown("---")
        st.markdown("### Скачать результат")
        
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            # Скачать в CSV
            result_df = pd.DataFrame([st.session_state.last_result])
            csv_result = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Скачать результат (CSV)",
                data=csv_result,
                file_name=f"task_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_download2:
            # Скачать в JSON
            json_result = json.dumps(st.session_state.last_result, ensure_ascii=False, indent=2).encode('utf-8')
            st.download_button(
                label="Скачать результат (JSON)",
                data=json_result,
                file_name=f"task_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

with tab2:
    st.markdown("### Загрузите файл с задачами")
    st.info("Поддерживаются форматы CSV и JSON. Обязательные колонки: наименование_задачи, описание_задачи")
    
    uploaded_file = st.file_uploader("Выберите файл", type=["csv", "json"])
    
    if uploaded_file is not None:
        # Определяем разделитель для CSV
        if uploaded_file.name.endswith('.csv'):
            # Пробуем разные разделители
            content = uploaded_file.getvalue().decode('utf-8')
            
            # Определяем разделитель
            if ';' in content.split('\n')[0]:
                sep = ';'
            elif ',' in content.split('\n')[0]:
                sep = ','
            elif '\t' in content.split('\n')[0]:
                sep = '\t'
            else:
                sep = ','
            
            # Читаем CSV с правильным разделителем
            df = pd.read_csv(uploaded_file, sep=sep)
            
            st.markdown("### Предпросмотр файла:")
            st.dataframe(df, use_container_width=True)
            st.caption(f"Всего строк: {len(df)}")
            
            # Сбрасываем указатель файла для последующей отправки
            uploaded_file.seek(0)
        else:
            # Для JSON файлов
            data = json.load(uploaded_file)
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                st.markdown("### Предпросмотр файла:")
                st.dataframe(df, use_container_width=True)
                st.caption(f"Всего записей: {len(data)}")
            else:
                st.warning("JSON файл должен содержать список задач")
            uploaded_file.seek(0)
        
        if st.button("Обработать файл", use_container_width=True):
            with st.spinner("Обработка файла..."):
                try:
                    # Для отправки нужно прочитать файл заново в байтах
                    if uploaded_file.name.endswith('.csv'):
                        uploaded_file.seek(0)
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    else:
                        uploaded_file.seek(0)
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/json")}
                    
                    response = requests.post("http://localhost:8000/task/batch", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get('results', [])
                        
                        # Делим оценку трудозатрат на 5 и приводим к верхнему регистру
                        for result in results:
                            if 'оценка_в_часах' in result:
                                result['оценка_в_часах'] = round(result['оценка_в_часах'] / 5, 1)
                            if 'категория_задачи' in result:
                                result['категория_задачи'] = result['категория_задачи'].capitalize()
                            if 'постановщик' in result:
                                result['постановщик'] = result['постановщик'].capitalize()
                            if 'улучшенное_наименование' in result and result['улучшенное_наименование']:
                                improved = result['улучшенное_наименование']
                                result['улучшенное_наименование'] = improved[0].upper() + improved[1:] if len(improved) > 1 else improved.upper()
                        
                        st.success(f"Обработано {len(results)} задач")
                        
                        # Отображение результатов
                        st.markdown("### Результаты обработки:")
                        df_results = pd.DataFrame(results)
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Кнопки скачивания результатов
                        st.markdown("### Скачать результаты:")
                        
                        col_down1, col_down2 = st.columns(2)
                        
                        with col_down1:
                            csv_output = df_results.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Скачать как CSV",
                                data=csv_output,
                                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        with col_down2:
                            json_output = json.dumps(results, ensure_ascii=False, indent=2).encode('utf-8')
                            st.download_button(
                                label="Скачать как JSON",
                                data=json_output,
                                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                    else:
                        error_detail = response.json().get('detail', 'Неизвестная ошибка')
                        st.error(f"Ошибка: {error_detail}")
                except Exception as e:
                    st.error(f"Не удалось подключиться к api: {e}")