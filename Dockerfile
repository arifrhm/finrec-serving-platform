FROM rayproject/ray:2.35.0-py310

WORKDIR /app

# Install system dependencies
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER ray

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000 8265

CMD ["serve", "run", "src.main:entrypoint"]
