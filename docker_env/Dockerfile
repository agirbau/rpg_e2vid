# --- Base image ---
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ENV no_proxy="10.81.0.0/16,172.16.0.0/16,localhost"
ENV http_proxy="http://10.81.247.64:8080"
ENV https_proxy="http://10.81.247.64:8080"

# Install HDF5 library, build tools, etc. from ROOT user
USER root

RUN apt-get update && apt-get install -y \
    libegl1 \
    libgl1 \
    libgomp1 \
    git \
    vim \
    wget \
    curl \
    htop \
    g++ \
    cmake \
    libhdf5-dev \
    ffmpeg \
    p7zip-full \
    && rm -rf /var/lib/apt/lists/*

# --- Create non-root user ---
# Create a user “user” with home directory /home/user
RUN useradd -ms /bin/bash user

# Create application directory and set ownership
RUN mkdir -p /home/user/app \
    && chown -R user:user /home/user/app

# Also ensure the workspace directory is owned
RUN mkdir -p /workspace \
    && chown -R user:user /workspace

# Add ~/.local/bin to PATH so pip user installs work
ENV PATH="/home/user/.local/bin:${PATH}"

# Switch to non-root user for subsequent commands
USER user
WORKDIR /home/user/app

# --- Copy project code into the container ---
# Use --chown to ensure correct file ownership
COPY --chown=user:user . /home/user/app

# --- Install Python / pip / dependencies ---
# Upgrade pip in user context
RUN pip install --no-cache-dir --upgrade pip

# Install required Python packages (numpy, pandas, opencv, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# --- Entrypoint / default command ---
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]