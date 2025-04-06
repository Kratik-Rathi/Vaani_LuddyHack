import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the keys from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
Huggingface_token = os.getenv("HF_TOKEN")
