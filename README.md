# HALOgen9000
# HALOgen9000 🖼️✨

## Overview

HALOgen9000 is an AI-powered image generation tool that creates unique images using Stable Diffusion and enhances prompts through intelligent processing. The project is designed to support content creation for the Instagram account [@halogen9000](https://www.instagram.com/halogen9000/).

## Features

- Generate high-quality images using Stable Diffusion XL
- Intelligent prompt enhancement with Ollama AI
- Support for GPU and CPU image generation
- Automatic prompt refinement and translation

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Diffusers
- Requests
- Ollama
- CUDA (optional, but recommended for faster generation)

## Installation

1. Clone the repository
2. Install required dependencies:
   ```bash
   pip install torch diffusers transformers requests
   ```
3. Ensure Ollama is running locally

## Usage

Generate an image with a simple command:

```bash
python generate.py "A school of flying fish in space"
```

### Example

```bash
# French input
python generate.py "Une bande de poisson volant dans l'espace"

# English translation of the prompt
# "A school of flying fish in space"
```

The script will:
- Generate an image
- Save the image as `generated_image.png`
- Create a refined prompt in `prompt.txt`

## Output

- `generated_image.png`: The generated image
- `prompt.txt`: Enhanced and translated prompt

## Note

Requires a local Ollama server running on `http://localhost:11434`


