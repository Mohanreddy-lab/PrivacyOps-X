FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY pyproject.toml README.md openenv.yaml models.py client.py __init__.py /app/
COPY server /app/server
COPY tasks /app/tasks
COPY outputs/evals /app/outputs/evals

RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.3" \
    "fastapi>=0.135.0" \
    "uvicorn>=0.43.0" \
    "openai>=2.26.0" \
    "httpx>=0.28.0" \
    "requests>=2.32.0" \
    "gradio>=4.0.0"

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
