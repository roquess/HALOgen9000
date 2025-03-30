import sys
import torch
import requests
import base64
import re
import json
import os
import uuid
from diffusers import StableDiffusionXLPipeline

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

def generate_image(prompt, num_steps=50, guidance_scale=7.5):
    """
    Generate image with Stable Diffusion
    """
    if torch.cuda.is_available():
        print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("CUDA not available. Using CPU (will be much slower)")
        device = "cpu"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True
    ).to(device)

    pipe.safety_checker = None

    image = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate.py \"Your prompt\"")
        sys.exit(1)

    prompt = sys.argv[1]

    try:
        # Ensure results directory exists
        results_dir = ensure_results_directory()
        
        # Generate a unique ID for this generation session
        unique_id = str(uuid.uuid4())
        
        # Set up file paths with matching names (different extensions)
        image_filename = os.path.join(results_dir, f"{unique_id}.png")
        prompt_filename = os.path.join(results_dir, f"{unique_id}.txt")

        # Generate the image
        image = generate_image(prompt)
        image.save(image_filename)
        print(f"Image saved: {image_filename}")

        # Use the generated image to generate an improved prompt
        generated_prompt = call_ollama(prompt, image_filename)
        print("Generated prompt:", generated_prompt)

        # Save the prompt
        with open(prompt_filename, "w", encoding="utf-8") as f:
            f.write(generated_prompt)
        print(f"Prompt saved: {prompt_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
