FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Устанавливаем pip и колеса
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Устанавливаем PyTorch отдельно (CPU версия для совместимости)
RUN pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Устанавливаем остальные зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем модели и код
COPY models/ /app/models/
COPY . .

RUN mkdir -p data/input data/output

EXPOSE 8000 8501

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]