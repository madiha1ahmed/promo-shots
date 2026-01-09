# Use a small, stable Python image
FROM python:3.11-slim

# Prevent Python from writing pyc files + enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create and switch to app directory
WORKDIR /app

# Install dependencies first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . /app

# Koyeb provides PORT at runtime
ENV PORT=8000

# Start with gunicorn (production server)
# NOTE: "app:app" means: file app.py, variable app = Flask(...)
CMD ["sh", "-c", "gunicorn -w 2 -k gthread --threads 4 --timeout 120 -b 0.0.0.0:${PORT} app:app"]
