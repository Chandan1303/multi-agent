FROM python:3.11-slim

WORKDIR /app

ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app:$PYTHONPATH"
# Tell uvicorn it's behind an HTTPS proxy (fixes Mixed Content on HF)
ENV FORWARDED_ALLOW_IPS="*"

# Install core dependencies from PyPI
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    openenv-core \
    transformers \
    accelerate \
    huggingface_hub \
    pydantic

# Install CPU-only torch separately (different index)
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

COPY . /app

EXPOSE 7860

# --proxy-headers tells uvicorn to trust X-Forwarded-Proto: https from HF's proxy
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips=*"]
