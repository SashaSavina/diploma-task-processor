#!/bin/bash

# запуск API
python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8000 &

# запуск Streamlit
streamlit run web/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true