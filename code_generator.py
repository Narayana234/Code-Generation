from models.model import code_model

def generate_code_from_text(prompt):
    return code_model.generate_code(prompt)

if __name__ == "__main__":
    test_prompt = "Write a Python function to check if a number is prime."
    print("Generated Code:\n", generate_code_from_text(test_prompt))
