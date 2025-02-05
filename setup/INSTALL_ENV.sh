echo "Setting up ECG LLM environment..."
echo "Using CUDA 11.8 and Python 3.11"

conda install -y pip

echo "Installing core libraries..."
conda install -y pandas numpy matplotlib seaborn tqdm ipykernel jupyter

echo "Installing data processing dependencies..."
conda install -y scikit-learn networkx scipy

echo "Installing PyTorch and CUDA..."
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia

echo "Installing NLP libraries..."
conda install -y -c conda-forge transformers tokenizers sentencepiece

echo "Installing Hugging Face libraries..."
pip install datasets accelerate huggingface_hub

echo "Installing ECG data processing libraries..."
pip install wfdb

echo "Installing LLM libraries..."
pip install langchain llama-index openai

echo "Installing API dependencies..."
pip install fastapi uvicorn

echo "Setting up Jupyter kernel..."
python -m ipykernel install --user --name="ecg_llm"

echo "Installation complete. Activate the environment with: conda activate ecg_llm"
