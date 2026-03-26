# Use a lightweight Python image
FROM python:3.10-slim

# Install system audio libraries (FFmpeg) required by Librosa
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your backend code (including models)
COPY . .

# Expose the port
EXPOSE 10000

# Run the production server with a 2-minute timeout for AI processing
CMD gunicorn -b 0.0.0.0:$PORT --timeout 120 app:app 