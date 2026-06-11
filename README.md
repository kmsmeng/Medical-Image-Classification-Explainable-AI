### Setup

1. Clone the repository
git clone https://github.com/kmsmeng/Medical-Image-Classification-Explainable-AI.git
cd Medical-Image-Classification-Explainable-AI

2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

3. Set up Pytorch
https://pytorch.org/get-started/locally/

To find out about the compute platform:
- go to kernel and type 'nvidia-smi'
- it will show something like 'CUDA Version: 13.0'. That is your Compute Platform.
- it will give you a command. run that in your terminal.

4. Install dependencies
pip install -r requirements.txt



5. To Download the Kaggle Dataset (NIH Chest X-rays)
- create a .env file
- get your 'username' and 'key' from the kaggle
- example .env file:
==========================================
KAGGLE_USERNAME=yourusername
KAGGLE_KEY=stringkey
==========================================

- after that the first code block will download the dataset locally in current working directory