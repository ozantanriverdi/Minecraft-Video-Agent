from transformers import AutoProcessor, AutoModelForCausalLM
import torch

class LLava_Model:
    def __init__(self, model_path="/home/atuin/v100dd/v100dd12/llava-v1.6-vicuna-7b", max_tokens=150):
        self.model_path = model_path
        self.max_tokens = max_tokens

        print(f"Loading LLaVA model from: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

    def forward(self, prompt, image, max_tries=5):
        attempt = 0
        while attempt < max_tries:
            try:
                if isinstance(image, str):
                    image = Image.open(image).convert("RGB")
                
                inputs = self.processor(prompt, images=image, return_tensors="pt").to(self.model.device)

                output = self.model.generate(**inputs, max_new_tokens=self.max_tokens)

                response = self.processor.decode(output[0], skip_special_tokens=True)

                return response
            
            except Exception as e:
                print(f"LLaVA inference failed (attempt {attempt + 1}): {e}")
                attempt += 1

        print("Max retries reached. Inference failed.")
        return None
