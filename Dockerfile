# Training/sampling image for diffusionlab.
#
# Default base includes CUDA + PyTorch; for a small CPU-only image:
#   docker build --build-arg BASE_IMAGE=python:3.11-slim -t diffusionlab:cpu .
ARG BASE_IMAGE=pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
FROM ${BASE_IMAGE}

# Never run ML workloads as root: the container touches mounted volumes.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# Install the package first for layer caching, then copy configs.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY configs ./configs

# Mount points for datasets and run artifacts.
RUN mkdir -p /app/data /app/runs && chown -R appuser:appuser /app
VOLUME ["/app/data", "/app/runs"]

USER appuser

ENTRYPOINT ["diffusionlab"]
CMD ["--version"]
