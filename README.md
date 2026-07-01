# Character-Level GPT Decoder

Educational implementation of a character-level Transformer decoder language model in PyTorch.

## Overview

This repository implements a small autoregressive language model inspired by GPT-style decoder blocks. It is intended for learning and demonstration: tokenization is character-level, training data is plain text, and the code mirrors common teaching implementations of self-attention, multi-head attention, feed-forward blocks, residual connections, and generation.

## Features

- Bigram baseline model in `bigram.py`
- Transformer decoder model in `GPT.py`
- Character-level tokenization from `input.txt`
- Training loop with periodic train/validation loss estimates
- Text generation from a zero-token context
- Learning notebooks preserved as experiment notes

## Tech Stack

- Python
- PyTorch
- Jupyter Notebook

## Architecture / Workflow

```text
input.txt -> character vocabulary -> train/validation tensors -> Transformer decoder blocks -> cross-entropy loss -> sampled text generation
```

## Setup Instructions

### Prerequisites

- Python 3.9+
- PyTorch installed for your CPU/GPU environment

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

No environment variables are required.

### Running the Project

```bash
python bigram.py
python GPT.py
```

## Example Usage

The scripts read `input.txt`, train a language model, and print generated character-level text. The generated text quality depends on training duration, compute, and the input corpus.

## Notebooks

- `GPT.ipynb`: exploratory notebook showing step-by-step implementation notes.
- `GPT_Running.ipynb`: run history and additional experiments. It contains notebook output noise and should be treated as exploratory.

## Repository Structure

```text
GPT.py             Transformer decoder implementation
bigram.py          Bigram baseline
input.txt          Training corpus
more.txt           Additional text notes/data
GPT.ipynb          Exploratory notebook
GPT_Running.ipynb  Run notebook with outputs
*.pdf              Reference material
```

## Limitations

- This is not a production LLM implementation.
- No pretrained weights are included.
- Hyperparameters are hard-coded in the scripts.
- Notebooks include exploratory output and should not be treated as the clean execution path.

## Future Improvements

- Add argparse-based configuration for dataset path, training steps, and output length.
- Save and load checkpoints.
- Add a minimal smoke test for model forward passes.
- Move exploratory notebooks under `notebooks/` after confirming links remain valid.

## Portfolio Note

This is an educational implementation intended to show understanding of decoder-only Transformer mechanics.
