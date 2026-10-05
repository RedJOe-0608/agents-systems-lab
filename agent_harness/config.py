import os

from dotenv import load_dotenv


load_dotenv()

API_KEY = os.environ["LLM_API_KEY"]
BASE_URL = os.environ["LLM_BASE_URL"]
MODEL_ID = os.environ["LLM_MODEL_ID"]
DATABASE_URL = os.environ["DATABASE_URL"]