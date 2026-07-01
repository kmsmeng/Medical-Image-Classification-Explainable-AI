### Setup

1. Clone the repository
```bash
git clone https://github.com/kmsmeng/Medical-Image-Classification-Explainable-AI.git
cd Medical-Image-Classification-Explainable-AI
```

2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

3. Set up Pytorch

Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) to get the install command for your system

To find out about the compute platform:
- go to kernel and type `nvidia-smi`
- it will show something like `CUDA Version: 13.0`. That is your Compute Platform.
- Use that version to select the right options on the PyTorch site, then run the install command it generates.

4. Install dependencies

```bash
pip install -r requirements.txt
```


5. To Download the Kaggle Dataset (NIH Chest X-rays)

- create a `.env` file in the project root
- get your `username` and `key` from kaggle
- example `.env` file:

```
KAGGLE_USERNAME=yourusername
KAGGLE_KEY=stringkey
```

- After that, the first code block in the notebook will download the dataset locally in current working directory