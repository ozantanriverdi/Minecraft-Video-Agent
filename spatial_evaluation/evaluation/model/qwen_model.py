from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


class Qwen_Model:
    def __init__(self, model_path="/home/atuin/v100dd/v100dd12/models/Qwen2.5-VL-7B-Instruct", max_tokens=100):
        self.model_path = model_path
        self.max_tokens = max_tokens

        print(f"Loading Qwen model from: {model_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def forward(self, prompt, image, max_tries=5):

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        attempt = 0
        while attempt < max_tries:
            try:
                
                text = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=False
                )

                image_inputs, video_inputs = process_vision_info(conversation)

                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")

                generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                print(output_text)
                return output_text
            
            except Exception as e:
                print(f"Qwen inference failed (attempt {attempt + 1}): {e}")
                attempt += 1

        print("Max retries reached. Inference failed.")
        return None