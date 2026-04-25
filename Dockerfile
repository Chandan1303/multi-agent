FROM python:3.11-slim

WORKDIR /app

ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app:$PYTHONPATH"

# Install core dependencies from PyPI
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    openenv-core \
    transformers \
    accelerate \
    pydantic

# Install CPU-only torch separately (different index)
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
