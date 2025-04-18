import sys
import torch
import requests
import base64
import re
import json
import os
import uuid
import argparse
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline

def safe_json_parse(response_text):
    """
    Safe and robust parsing of JSON responses from Ollama

    Args:
    - response_text (str): Raw response text

    Returns:
    - str: Extracted content or original text
    """
    try:
        # Attempt direct JSON parsing
        response_json = json.loads(response_text)
        if isinstance(response_json, dict) and 'message' in response_json:
            content = response_json['message'].get('content', '')
            return content
    except json.JSONDecodeError:
        pass

    # Alternative extraction methods
    extraction_methods = [
        r'"content"\s*:\s*"([^"]+)"',  # Extract content between quotes
        r'"message"\s*:\s*{\s*"content"\s*:\s*"([^"]+)"',  # Extract message content
        r'content"\s*:\s*"([^"]+)"'  # Another variant
    ]

    for pattern in extraction_methods:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            content = match.group(1)
            return content

    # Fallback: return cleaned raw text
    cleaned_text = re.sub(r'\s+', ' ', response_text).strip()
    return cleaned_text

def extract_prompt_from_ollama_response(response):
    """
    Extract image generation prompt
    """
    # Patterns to extract the prompt
    prompt_patterns = [
        r'Stable Diffusion Prompt:[\s]*(.+)',
        r'Prompt:[\s]*(.+)',
        r'Image Generation Prompt:[\s]*(.+)'
    ]

    for pattern in prompt_patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return response.strip()

def call_ollama(prompt, image_path=None, model="llava:13b"):
    """
    Call Ollama for prompt generation and translation
    """
    url = "http://localhost:11434/api/chat"

    # Payload for prompt generation
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant specialized in generating prompts for Stable Diffusion. "
                           "From a given prompt, generate a detailed, creative, and precise prompt in English. "
                           "IMPORTANT : generate representing Instagram tags and a rich visual description."
            },
            {
                "role": "user",
                "content": f"Generate a creative Stable Diffusion prompt from: {prompt}"
            }
        ]
    }

    if image_path:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        payload["messages"].append({
            "role": "user",
            "content": "Consider this image as a reference",
            "images": [encoded_image]
        })

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        # Safe parsing of content
        full_response_content = safe_json_parse(response.text)
        generated_prompt = extract_prompt_from_ollama_response(full_response_content)

        # Translation to English
        translation_payload = {
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a translator. Translate the following text to English creatively."
                },
                {
                    "role": "user",
                    "content": generated_prompt
                }
            ]
        }

        translation_response = requests.post(url, json=translation_payload)
        translation_response.raise_for_status()

        # Safe parsing of translation
        translated_content = safe_json_parse(translation_response.text)
        return translated_content.strip() or generated_prompt

    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return prompt

def get_sd_model_info(model_name):
    """
    Get model configuration based on model name
    """
    models = {
        "sdxl": {
            "class": StableDiffusionXLPipeline,
            "path": "stabilityai/stable-diffusion-xl-base-1.0",
            "default_size": (1024, 1024)
        },
        "sd15": {
            "class": StableDiffusionPipeline,
            "path": "runwayml/stable-diffusion-v1-5",
            "default_size": (512, 512)
        },
        "sd21": {
            "class": StableDiffusionPipeline,
            "path": "stabilityai/stable-diffusion-2-1",
            "default_size": (768, 768)
        },
        "dreamshaper": {
            "class": StableDiffusionPipeline, 
            "path": "Lykon/dreamshaper-7",
            "default_size": (512, 512)
        }
    }
    
    if model_name not in models:
        print(f"Unknown model: {model_name}. Using SDXL as default.")
        return models["sdxl"]
    
    return models[model_name]

def generate_image(prompt, num_steps=50, guidance_scale=7.5, width=1024, height=1024, sd_model="sdxl"):
    """
    Generate image with Stable Diffusion
    """
    if torch.cuda.is_available():
        print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("CUDA not available. Using CPU (will be much slower)")
        device = "cpu"

    # Get model configuration
    model_info = get_sd_model_info(sd_model)
    
    # Memory optimization for smaller models
    if device == "cuda" and sd_model != "sdxl":
        # Enable attention slicing for memory efficiency
        with torch.cuda.amp.autocast():
            pipe = model_info["class"].from_pretrained(
                model_info["path"],
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to(device)
            pipe.enable_attention_slicing()
    else:
        pipe = model_info["class"].from_pretrained(
            model_info["path"],
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True
        ).to(device)

    pipe.safety_checker = None

    image = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height
    ).images[0]

    return image

def ensure_results_directory():
    """
    Create results directory if it doesn't exist
    """
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def parse_size(size_str, sd_model="sdxl"):
    """
    Parse the size string into width and height
    """
    # Get default size for the selected model
    default_size = get_sd_model_info(sd_model)["default_size"]
    
    sizes = {
        "square": default_size,
        "portrait": (default_size[0], int(default_size[1] * 1.25)),
        "landscape": (int(default_size[0] * 1.25), default_size[1]),
        "wide": (int(default_size[0] * 1.5), int(default_size[1] * 0.875)),
        "tall": (int(default_size[0] * 0.875), int(default_size[1] * 1.5)),
        "cinematic": (int(default_size[0] * 1.875), int(default_size[1] * 0.8125)),
        "default": default_size
    }

    # Check if it's a predefined size
    if size_str.lower() in sizes:
        return sizes[size_str.lower()]

    # Check if it's a custom size in "widthxheight" format
    if "x" in size_str:
        try:
            width, height = map(int, size_str.split("x"))
            # Ensure dimensions are valid (multiples of 8)
            width = max(256, (width // 8) * 8)
            height = max(256, (height // 8) * 8)
            return width, height
        except ValueError:
            pass

    # Return default size if parsing fails
    print(f"Invalid size format: {size_str}. Using default size for {sd_model}: {default_size[0]}x{default_size[1]}.")
    return default_size

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion")
    parser.add_argument("prompt", help="Input prompt for image generation")
    parser.add_argument("-o", "--ollama", type=int, default=1, choices=[0, 1],
                        help="Use Ollama for prompt enhancement: 1=yes (default), 0=no")
    parser.add_argument("-s", "--steps", type=int, default=50,
                        help="Number of inference steps (default: 50)")
    parser.add_argument("-g", "--guidance", type=float, default=7.5,
                        help="Guidance scale (default: 7.5)")
    parser.add_argument("-z", "--size", type=str, default="square",
                        help="Image size: 'square', 'portrait', 'landscape', 'wide', 'tall', 'cinematic', " +
                             "or custom dimensions as 'widthxheight' (e.g., '512x512')")
    parser.add_argument("-m", "--model", type=str, default="sdxl", choices=["sdxl", "sd15", "sd21", "dreamshaper"],
                        help="Stable Diffusion model: 'sdxl' (default), 'sd15' (SD 1.5), 'sd21' (SD 2.1), 'dreamshaper'")

    args = parser.parse_args()

    try:
        # Ensure results directory exists
        results_dir = ensure_results_directory()

        # Generate a unique ID for this generation session
        unique_id = str(uuid.uuid4())

        # Set up file paths with matching names (different extensions)
        image_filename = os.path.join(results_dir, f"{unique_id}.png")

        # Parse image size based on selected model
        width, height = parse_size(args.size, args.model)
        print(f"Using model: {args.model}")
        print(f"Using image dimensions: {width}x{height}")

        # Generate the image using the original prompt and specified size
        print(f"Generating image for prompt: {args.prompt}")
        image = generate_image(args.prompt, args.steps, args.guidance, width, height, args.model)
        image.save(image_filename)
        print(f"Image saved: {image_filename}")

        # Only use Ollama if the flag is set to 1
        if args.ollama == 1:
            # Use the generated image to generate an improved prompt
            generated_prompt = call_ollama(args.prompt, image_filename)
            print("Generated prompt:", generated_prompt)

            # Save the prompt
            prompt_filename = os.path.join(results_dir, f"{unique_id}.txt")
            with open(prompt_filename, "w", encoding="utf-8") as f:
                f.write(generated_prompt)
            print(f"Prompt saved: {prompt_filename}")
        else:
            print("Ollama prompt enhancement skipped.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
