FROM python:3.10-slim

WORKDIR /app

# Install system dependencies and a text editor (nano)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        nano && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy application files
COPY ./src ./src
COPY requirements.txt ./requirements.txt

# Create a default .env file
RUN touch .env

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose port 8501 for Streamlit
EXPOSE 8501

# Set the default command to run Streamlit
CMD ["streamlit", "run", "src/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
