import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import settings

_gemini_client = None

def get_gemini_client():
    global _gemini_client
    if _gemini_client is None and settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "mock_key_for_testing":
        try:
            from google import genai
            _gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
        except Exception:
            _gemini_client = None
    return _gemini_client

def multimodal_verify_leaf(image_input) -> Optional[Dict[str, Any]]:
    """
    Multimodal visual verification using Google Gemini Vision.
    Corrects misidentifications for crops not in offline 15-class classifier (like Rice).
    """
    client = get_gemini_client()
    if client is None:
        return None

    try:
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            return None

        prompt = """You are a senior agricultural botanist and plant pathologist.
Examine this plant leaf photo carefully. Identify the exact crop species and any visible disease.
Return ONLY a valid JSON markdown block:
```json
{
  "crop": "<Crop name, e.g. Rice, Wheat, Tomato, Potato, Maize, Apple, Grape, Pepper, etc.>",
  "disease": "<Disease name, e.g. Brown Spot, Rice Blast, Bacterial Leaf Blight, Stripe Rust, Late Blight, Healthy, etc.>",
  "confidence": 95.0,
  "is_leaf": true,
  "symptoms": "<Brief description of foliar lesions>"
}
```"""

        candidate_models = [settings.GEMINI_MODEL, "gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-2.5-flash"]
        resp = None
        for m in candidate_models:
            if not m:
                continue
            try:
                resp = client.models.generate_content(
                    model=m,
                    contents=[pil_img, prompt]
                )
                if resp and resp.text:
                    break
            except Exception:
                continue

        if resp is None or not resp.text:
            return None
        text = resp.text.strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            return data
        # Fallback if raw json without markdown
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)
    except Exception as exc:
        print(f"Multimodal verifier error: {exc}")
        return None

    return None

