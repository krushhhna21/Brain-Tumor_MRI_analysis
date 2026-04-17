FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download the model during build
RUN python download_model.py

EXPOSE 5000

CMD ["gunicorn", "app:app"]
