import os
from openai import OpenAI
from os.path import join

class Spatial_Agent:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
        client = OpenAI(api_key=self.api_key)

    def default_gpt_agent(self):
        pass

    def socratic_gpt_agent(self):
        pass