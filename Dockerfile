# Use Ubuntu 22.04 for the most stable SUMO PPA support
FROM ubuntu:22.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies and SUMO
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    && add-apt-repository ppa:sumo/stable \
    && apt-get update && apt-get install -y \
    sumo sumo-tools sumo-doc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up SUMO_HOME
ENV SUMO_HOME=/usr/share/sumo

# Set the working directory
WORKDIR /app

# Copy requirement file and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the backend source code
# Note: .dockerignore handles excluding the frontend and local artifacts
COPY . .

# Ensure the 'src' directory is in the Python path
ENV PYTHONPATH=/app/src

# Render provides the port in the $PORT environment variable
# Our run_server.py is already updated to handle this.
EXPOSE 8000

# Start the server
CMD ["python3", "run_server.py"]
