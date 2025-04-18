# Use Python 3.11 slim as the base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create directories for data, notebooks, and logs
RUN mkdir -p /app/data /app/notebooks /app/logs

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install specific version of pandas and other key libraries
RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    numpy \
    matplotlib \
    plotly \
    networkx \
    neo4j \
    jupyter \
    jupyterlab \
    ipywidgets \
    scikit-learn \
    pytest \
    black \
    flake8

# Copy requirements and install additional dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user
RUN useradd -m -u 1000 appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose ports for Jupyter and other services
EXPOSE 8888

# Set the entrypoint
CMD ["bash"] 