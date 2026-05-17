FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1


WORKDIR /yournewface

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install \
        --timeout 1000 \
        --retries 5 \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        -r requirements.txt
COPY . .
EXPOSE 8000


# CMD 
# CMD ["python", "app.py"]