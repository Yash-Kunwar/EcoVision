import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("API Key not found! Make sure .env file is correct.")

# 2. Configure Gemini
genai.configure(api_key=api_key)

# We use the faster, cheaper model for text tasks
model = genai.GenerativeModel('gemini-2.5-flash')

def fetch_animal_info(animal_label):
    """
    Takes an animal label (e.g., 'spider') and asks Gemini for 
    structured facts and scientific data.
    """
    print(f" Did you know...")

    # A strict prompt to force JSON output
    prompt = f"""
    You are a zoologist API. I will give you an animal name. 
    You must return a valid JSON object with the following fields:
    - "common_name": The capitalized common name.
    - "scientific_name": The Latin scientific name.
    - "fun_facts": A list of 3 short, interesting facts (max 15 words each).
    - "genus_members": A list of 3 other animals that belong to the same Genus (just names).
    
    The animal is: {animal_label}
    
    IMPORTANT: Return ONLY the JSON. No markdown formatting (like ```json), no intro text.
    """

    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        
        # Clean up if Gemini accidentally adds markdown code blocks
        if raw_text.startswith("```"):
            raw_text = raw_text.replace("```json", "").replace("```", "")
            
        # Parse text into a Python Dictionary
        data = json.loads(raw_text)
        return data

    except Exception as e:
        print(f"❌ GenAI Error: {e}")
        # Fallback data so the app doesn't crash
        return {
            "common_name": animal_label.capitalize(),
            "scientific_name": "Unknown",
            "fun_facts": ["Could not retrieve facts at this time."],
            "genus_members": []
        }

# This allows us to test this file directly
if __name__ == "__main__":
    # Test Run
    result = fetch_animal_info("spider")
    print(json.dumps(result, indent=2))