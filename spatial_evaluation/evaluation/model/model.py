import os

from .gpt_model import GPT_Model
from .gpt_socratic import GPT_Socratic_Model

class Model:
    def __init__(self, model_type):
        self.model_type = model_type
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
        if self.model_type == "gpt":
            self.model = GPT_Model(self.api_key)
        elif self.model_type == "gpt_socratic":
            self.model = GPT_Socratic_Model(self.api_key)

    def forward(self, prompt, image):
        # if self.model_type == "gpt":
        #     return self.model.forward(prompt, image)
        # elif self.model_type == "gpt_socratic":
        return self.model.forward(prompt, image)
