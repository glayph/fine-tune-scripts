```markdown
# Local LLM Fine-Tune Scripts

A lightweight, streamlined toolset for fine-tuning Large Language Models (LLMs) locally. This repository provides execution scripts and configuration files to train and customize models using your own datasets and system prompts.

---

## 📁 Repository Structure

```text
.
├── Prompt.md              # Template and guidelines for prompt formats
├── requirements.txt       # Python packages and dependencies
├── start_finetune.bat     # Windows batch script to start fine-tuning with one click
├── system_prompt.txt      # System prompt configuration file
└── train.py               # Main fine-tuning pipeline script

```

---

## ✨ Features

* **Local Execution:** Fine-tune open-weight LLMs locally on your own hardware.
* **One-Click Execution:** Run `start_finetune.bat` on Windows to quickly trigger the training pipeline.
* **Custom System Instructions:** Easily modify model behavior using `system_prompt.txt`.
* **Modular Setup:** Clean separation of dependencies, prompts, and training logic.

---

## ⚙️ Installation & Requirements

### 1. Prerequisites

* **Python 3.9+**
* **NVIDIA GPU** with CUDA support recommended for hardware acceleration.

### 2. Setup

Clone the repository and install the necessary dependencies:

```bash
# Clone the repository
git clone [https://github.com/glayph/fine-tune-scripts.git](https://github.com/glayph/fine-tune-scripts.git)
cd fine-tune-scripts

# Install dependencies
pip install -r requirements.txt

```

---

## 🚀 How to Use

### 1. Configure System Prompt & Dataset

* Edit **`system_prompt.txt`** to define the standard persona or instructions for your model.
* Check **`Prompt.md`** for recommendations on structuring your training prompt templates.

### 2. Run Fine-Tuning

**On Windows:**
Simply run the batch script from the terminal or double-click it:

```cmd
start_finetune.bat

```

**On Linux / macOS / Terminal:**
Execute the training script directly:

```bash
python train.py

```

---

## 📄 File Overview

| File Name | Purpose |
| --- | --- |
| **`train.py`** | Loads the model, processes dataset inputs, manages hyperparameters, and runs the training loop. |
| **`start_finetune.bat`** | Automation script for Windows users to launch the fine-tuning process. |
| **`requirements.txt`** | Contains Python dependencies (PyTorch, Transformers, PEFT/LoRA, Datasets, etc.). |
| **`system_prompt.txt`** | Stores the core system instructions applied during training/inference. |
| **`Prompt.md`** | Documentation on training dataset formatting and prompt template structures. |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

```
I Fucked the LiCeNSe'S
```
