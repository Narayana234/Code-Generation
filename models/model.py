from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class CodeT5Model:
    def __init__(self, model_name="Salesforce/codet5-small"):
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model: {e}")

    def generate_code(self, prompt, max_length=100):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = self.model.generate(**inputs, max_length=max_length, num_return_sequences=1)
            generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_code
        except Exception as e:
            print(f"Error during code generation: {e}")
            return "Error generating code"

# Initialize model
code_model = CodeT5Model()
