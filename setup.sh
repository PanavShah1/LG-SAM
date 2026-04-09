#! /bin/bash

echo "Creating virtual environment..."
# Create a virtual environment
./bin/uv venv
source .venv/bin/activate

echo "Installing dependencies..."
# Install dependencies
./bin/uv pip install -r requirements.txt

mkdir vendored

echo 'Installing `sam3`...'
# Install `sam3`
git clone https://github.com/facebookresearch/sam3.git vendored/sam3 # Clone `sam3`
cd vendored/sam3
sed -i 's/numpy==1.24.0/#numpy==1.24.0/' pyproject.toml # Patch `numpy` in pyproject.toml
../../bin/uv pip install -e . # Install `sam3`
cd ../../

echo "Extracting checkpoints..."
# Extract checkpoints.zip
unzip checkpoints.zip

echo "Downloading open-sourced models..."
# SAM3, RemoteSAM, EarthMind are already included in `checkpoints.zip`
# Download open-sourced models
cd checkpoints
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir Qwen3-VL-8B-Instruct
hf download Qwen/Qwen-Image-Edit-2509 --local-dir Qwen-Image-Edit-2509
hf download google-bert/bert-base-uncased --local-dir bert-base-uncased
rm -rf **/.cache
cd ..

# export PYTHON_ENV=prod

# echo "Starting the server..."
# # Run `app.py`
# ./bin/uv run uvicorn app:app --host 0.0.0.0 --port 8001
echo "creating server config..."
sudo bash -c 'cat > /etc/supervisor/conf.d/lgsam.conf << '\''EOF'\''
[program:lgsam]
command=/datasets/dataset-final/ISRO-lg-sam/.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8001
directory=/datasets/dataset-final/ISRO-lg-sam
user=root
autostart=true
autorestart=true
stdout_logfile=/var/log/isro-lg-sam.log
stderr_logfile=/var/log/isro-lg-sam.log
environment=PATH="/datasets/dataset-final/ISRO-lg-sam/.venv/bin:/usr/local/cuda/bin:/usr/local/cuda-12.4/bin:%(ENV_PATH)s",LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:/usr/local/cuda/lib64",CUDA_HOME="/usr/local/cuda-12.4",MKL_THREADING_LAYER="GNU",PYTHON_ENV="prod"
EOF'

echo "restarting supervisor..."
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl restart lgsam
echo "server config done..."