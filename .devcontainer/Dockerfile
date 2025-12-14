FROM nvcr.io/nvidia/pytorch:25.09-py3

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -sf /root/.local/bin/uv /usr/local/bin/uv

ENV PATH="/root/.local/bin:${PATH}"

# Helpful defaults for Python inside containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
