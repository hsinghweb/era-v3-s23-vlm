# ERA-V3 Session 23: Vision Language Model Implementation

This repository implements a Vision Language Model (VLM) that combines SigLIP vision encoder with Phi-3 text decoder using QLoRA optimization for efficient training and deployment.

## Model Architecture

- **Vision Encoder**: SigLIP model for image feature extraction
- **Text Decoder**: Microsoft's Phi-3 model with frozen parameters
- **Optimization**: QLoRA (Quantized Low-Rank Adaptation) for memory-efficient fine-tuning
- **Projection Layer**: Linear projection from vision embeddings (512-dim) to text model dimension

## Key Features

- 4-bit quantization for reduced memory footprint
- Parameter-efficient fine-tuning using LoRA
- Custom dataset handling for image-text pairs
- Optimized for Google Colab execution

## Setup and Dependencies

```bash
pip install torch torchvision transformers tqdm matplotlib peft bitsandbytes accelerate wandb
```

## Training Process

1. **Data Preparation**:
   - Custom dataset implementation for image-text pairs
   - Automatic parsing of image descriptions
   - Image transformation pipeline

2. **Model Initialization**:
   - Load and quantize vision encoder (SigLIP)
   - Initialize text decoder (Phi-3) with 4-bit quantization
   - Apply QLoRA optimization
   - Freeze base model parameters

3. **Training**:
   - Combined training of vision-language model
   - Logging with WandB integration
   - Parameter-efficient updates through LoRA layers

## Usage

The implementation is provided in the `s23.ipynb` notebook, which can be run directly in Google Colab. The notebook includes detailed implementation steps and training procedures.

## Acknowledgments

- SigLIP vision encoder
- Microsoft Phi-3 language model
- QLoRA optimization technique
