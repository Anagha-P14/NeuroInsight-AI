FROM python:3.11-slim[cite: 1]

WORKDIR /app[cite: 1]

COPY requirements.txt .[cite: 1]
RUN pip install --no-cache-dir -r requirements.txt[cite: 1]

COPY main.py .[cite: 1]

# Fallback environment variable
ENV PORT=8080[cite: 1]

# Use shell execution syntax so ${PORT} resolves properly at runtime
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
