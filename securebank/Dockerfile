# Use the official lightweight Python 3.9 slim image as the base.
FROM python:3.9-slim

# Set environment variables to:
# - avoid writing .pyc files,
# - ensure stdout/stderr are unbuffered (helpful for logging),
# - configure Flask app environment variables for running the app.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_ENV=production

# Set working directory inside the container.
WORKDIR /app/securebank

# Install OS-level dependencies needed for building packages and PostgreSQL client libraries.
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*  # Clean up apt cache to reduce image size

# Copy requirements.txt first for dependency installation.
COPY requirements.txt /app/securebank/requirements.txt

# Upgrade pip and install Python dependencies without cache to reduce image size.
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the entire project source code into the container, but exclude data_sources.
# (ensure you have a .dockerignore that excludes securebank/data_sources/* if needed)
COPY . /app/securebank

# Create directories for storing models, datasets, logs, and an empty data_sources folder.
RUN mkdir -p storage/models storage/datasets logs data_sources

# Expose port 5000 for Flask app to listen on.
EXPOSE 5000

# Set PYTHONPATH to ensure imports work correctly inside the container.
ENV PYTHONPATH=/app/securebank

# Run the Flask application, binding it to all network interfaces.
CMD ["flask", "run", "--host=0.0.0.0"]