FROM python:3.13.5-bullseye

COPY requirements.txt /app/
COPY . /app/

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
