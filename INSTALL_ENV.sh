#cuda 12.4, python 11

#cuda 12.4, python 11
conda install -y pip

#stl
conda install -y pandas
conda install -y numpy
conda install -y matplotlib
conda install -y seaborn
conda install -y tqdm
conda install -y ipykernel
conda install -y jupyter

#data processing
conda install -y scikit-learn
conda install -y networkx
conda install -y scipy

#kernel
python -m ipykernel install --user --name="ecg_llm"

#ai stack (torch = 2.2, cuda = 11.8, pyg = 2.5.0) --> compatible with cuda 12.2
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.2 -c pytorch -c nvidia

conda install -y -c conda-forge transformers tokenizers

pip install openai

conda install -y -c conda-forge sentencepiece

pip install langchain
pip install fastapi uvicorn
pip install llama-index
