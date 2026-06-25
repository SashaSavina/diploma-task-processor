from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import json
import os
import tempfile
import uvicorn

from src.processor import TaskProcessor
from src.file_processor import FileProcessor

# инициализация процессоров
processor = TaskProcessor(models_path='./models')
file_processor = FileProcessor(processor)

app = FastAPI(
    title="Task Processing API",
    description="API для обработки ИТ-задач: классификация, оценка трудозатрат, улучшение формулировок",
    version="1.0.0"
)

# модели данных для API
class TaskRequest(BaseModel):
    наименование_задачи: str
    описание_задачи: str
    постановщик: Optional[str] = None
    приоритет: Optional[str] = None
    комментарии: Optional[str] = ""
    дата_постановки: Optional[str] = None

class TaskResponse(BaseModel):
    улучшенное_наименование: str
    исходное_описание: str
    категория_задачи: str
    оценка_в_часах: float
    постановщик: str
    приоритет: str
    дата_обработки: str

# эндпоинты
@app.get("/")
async def root():
    return {"message": "Task Processing API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/task", response_model=TaskResponse)
async def process_single_task(request: TaskRequest):
    """Обработка одной задачи"""
    try:
        task_data = request.dict()
        print(f"DEBUG API: получен task_data = {task_data}")  # <- добавить
        result = processor.process_task(task_data)
        return TaskResponse(**result)
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Ошибка при обработке задачи: {error_detail}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/task/batch")
async def process_batch_tasks(file: UploadFile = File(...)):
    """Обработка пакета задач из файла"""
    try:
        # сохраняем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # определяем формат и обрабатываем
        if file.filename.endswith('.csv'):
            results = file_processor.process_csv(tmp_path)
        elif file.filename.endswith('.json'):
            results = file_processor.process_json(tmp_path)
        else:
            raise ValueError("Поддерживаются только CSV и JSON файлы")
        
        # удаляем временный файл
        os.unlink(tmp_path)
        
        return {"status": "success", "total": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/task/batch/upload")
async def upload_and_process(file: UploadFile = File(...), output_format: str = "json"):
    """Загрузка файла, обработка и возврат результата"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        if file.filename.endswith('.csv'):
            results = file_processor.process_csv(tmp_path)
        elif file.filename.endswith('.json'):
            results = file_processor.process_json(tmp_path)
        else:
            raise ValueError("Поддерживаются только CSV и JSON файлы")
        
        os.unlink(tmp_path)
        
        # возвращаем результаты в нужном формате
        if output_format == "csv":
            df = pd.DataFrame(results)
            csv_output = df.to_csv(index=False)
            return JSONResponse(content={"csv": csv_output})
        else:
            return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def run_api(host="0.0.0.0", port=8000):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_api()