FROM python:3.12-slim

WORKDIR /app
ENV EBUS_ROOT=/app PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY data/instances ./data/instances
COPY data/hamburg_winter_temps.csv ./data/hamburg_winter_temps.csv

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "ebus.service:app", "--host", "0.0.0.0", "--port", "8000"]
