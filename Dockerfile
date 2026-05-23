FROM python:3.12-slim

WORKDIR /app

# opencv-python-headless still needs a couple of shared libs for image codecs
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sidecar.py .
COPY classifier/ ./classifier/

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "sidecar.py"]
