import os
from dotenv import load_dotenv

from .gpt_model import GPT_Model
from .gpt_socratic import GPT_Socratic_Model
from .llava_model import LLava_Model
from .qwen_model import Qwen_Model

class Model:
    def __init__(self, model_type):
        self.model_type = model_type
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
        if self.model_type == "gpt":
            self.model = GPT_Model(self.api_key)
        elif self.model_type == "gpt_socratic":
            self.model = GPT_Socratic_Model(self.api_key)
        elif self.model_type == "llava":
            self.model = LLava_Model()
        elif self.model_type == "qwen":
            self.model = Qwen_Model()

    def forward(self, prompt, image):
        if self.model_type == "gpt_socratic":
            output, socratic_description = self.model.forward(prompt, image)
            return output, socratic_description
        else:
            output = self.model.forward(prompt, image)
            return output, None
