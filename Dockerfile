FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir pandas==2.2.2 \
    numpy \
    tqdm \
    python-dateutil

# Copy the processing script and data
COPY process_bookmarks.py /app/
COPY data-bookmark.jsonl /app/data-bookmark.jsonl

# Make directory for output
RUN mkdir -p /app/output

# Set permissions
RUN chmod +x /app/process_bookmarks.py

ENTRYPOINT ["python", "/app/process_bookmarks.py"]

# Default arguments - can be overridden at runtime
CMD ["--input", "/app/data-bookmark.jsonl", "--output", "/app/output/organized-bookmarks.jsonl", "--pretty", "--group-by", "source"] 