import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

class LLava_Model:
    def __init__(self, model_path="/home/atuin/v100dd/v100dd12/llava-v1.6-vicuna-7b", max_tokens=100):
        self.model_path = model_path
        self.max_tokens = max_tokens

        print(f"Loading LLaVA model from: {model_path}")
        self.model = LlavaForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=torch.float16, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def forward(self, prompt, image, max_tries=5):

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        attempt = 0
        while attempt < max_tries:
            try:
                
                inputs = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.model.device, torch.float16)

                generate_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
                response = self.processor.batch_decode(generate_ids, skip_special_tokens=True)

                return response
            
            except Exception as e:
                print(f"LLaVA inference failed (attempt {attempt + 1}): {e}")
                attempt += 1

        print("Max retries reached. Inference failed.")
        return None
