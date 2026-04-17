# RLSTC Setup Guide (README2.md)

This document describes the steps taken to set up the RLSTC (Sub-trajectory Clustering with Deep Reinforcement Learning) project for execution, and provides instructions for others to recreate the environment.

## What Was Done

1. **Environment Setup**:
   - Installed Python 3.6 using the system package manager (apt) on Ubuntu 24.04.
   - Created a virtual environment named `.venv36` using Python 3.6.
   - Activated the virtual environment.

2. **Dependency Installation**:
   - Installed the required packages: TensorFlow 2.2.0, Keras 2.4.3, scikit-learn 0.24.2, tqdm 4.62.0, numpy 1.19.2.
   - Note: TensorFlow 2.2.0 requires Python 3.6-3.8; later versions are not compatible.

3. **Data and Models**:
   - Extracted `savemodels.tar.gz` to obtain pre-trained models.
   - Data directories (e.g., `data/`) contain the necessary datasets as per the original README.

4. **Version Control**:
   - Created a `.gitignore` file to exclude large files (data, models, virtual environments) from git tracking.

## How to Recreate This Setup

Follow these steps on a Linux system (Ubuntu recommended) to set up the environment identically.

### 1. Install pyenv (for managing Python versions)
```bash
curl https://pyenv.run | bash
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc
```

### 2. Install Python 3.6 with pyenv
```bash
pyenv install 3.6.15  # Use a stable 3.6 version
pyenv global 3.6.15
```

### 3. Create and Activate Virtual Environment
```bash
python -m venv .venv36
source .venv36/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Extract Data and Models
```bash
tar -xzvf data.tar.gz  # If downloaded from the provided link
tar -xzvf savemodels.tar.gz  # If downloaded from the provided link
```

### 6. Run the Code
- Navigate to `RLSTCcode/subtrajcluster/`
- Run preprocessing: `python preprocessing.py`
- Run training: `python rl_train.py`
- For cross-validation: `python crosstrain.py`

## Notes
- Training can take a long time; consider running overnight.
- If using a different OS, adjust the Python installation method accordingly.
- Ensure the virtual environment is activated before running any Python commands.
- The `.gitignore` file prevents large files from being committed; if needed, use Git LFS for tracking large files instead.

## Troubleshooting
- If TensorFlow installation fails, ensure Python 3.6 is used (check with `python --version`).
- For GPU support, install TensorFlow with GPU dependencies if CUDA is available.
- If packages conflict, recreate the virtual environment and reinstall.