# Conformer Encoder Pretraining with Best-RQ Algorithm

This repository provides the code to pretrain a 120M Conformer encoder using the **Best-RQ** algorithm. The Best-RQ approach offers a simpler and more compute-efficient pretraining strategy for audio speech language models compared to contrastive learning methods like **wav2vec2**. This implementation is built using **PyTorch Lightning** and is designed to be easily extensible for different datasets and configurations.

## Features
- **Pretraining Approach**: Leverages the Best-RQ algorithm for pretraining.
- **Multilingual LibriSpeech**: Trains on the Multilingual LibriSpeech dataset downloaded from Hugging Face by default.
- **Customizable Datasets**: To train on a different dataset, simply modify the `dataset.py` file.
- **Logging**: Uses Weights & Biases (WandB) for experiment tracking. The training script prompts you with the necessary setup steps.

## Installation and Usage

1. **Clone this repository**:
   ```bash
   git clone <REPO_URL>
   cd <REPO_NAME>
2. **Run the training script**:
    ```bash
    pip install -r requirements.txt
    python3 train.py

## Model Architecture

The Conformer model in this repository features **relational positional encoding attention**, powered by the latest **FlexAttention API** from PyTorch 2.5.1, replacing traditional absolute positional encodings and vanilla attention. This implementation includes modifications to the standard Conformer model, but future improvements aim to explore newer advancements like **grouped query attention**.

### Future Work

- **Conformer Architecture Revisions**: Rewriting and refining the Conformer model, drawing inspiration from more recent research like **Efficient Conformer**.
- **Pretraining an RNN-T ASR Model**: Initializing with the pretrained encoder and training a **Conformer RNN-T** model, inspired by **AssemblyAI's research**.
- **Best-RQ Algorithm Article**: An in-depth article on the Best-RQ algorithm is in progress.

## References

- **Self-Supervised Learning with Random-Projection Quantizer for Speech Recognition**  
  [Link to Paper](https://arxiv.org/pdf/2202.01855)
- **Conformer: Convolution-augmented Transformer for Speech Recognition**  
  [Link to Paper](https://arxiv.org/pdf/2005.08100)
- **FlexAttention in PyTorch**  
  [Link to Blog](https://pytorch.org/blog/flexattention/)

## Questions?

If you have any questions or need further clarification, please feel free to reach out!
