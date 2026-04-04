FROM python:3.12-slim


RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /yournewface

# requirements.txt (кэш!)
COPY requirements.txt /yournewface/

# 
RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
    --timeout 1000 \
    --retries 5 \
    # --extra-index-url https://download.pytorch.org/whl/cu130 \
    -r /yournewface/requirements.txt

COPY . /yournewface/
WORKDIR /yournewface

EXPOSE 8080

# CMD 
# CMD ["python", "app.py"]