import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio
import sys
import io
import contextlib
import whisper
import tempfile
import ffmpeg
import time
from functools import lru_cache
from streamlit_mic_recorder import mic_recorder

# Fix for Windows Asyncio Issue
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception as e:
        st.error(f"Error setting asyncio policy: {e}")

# Set Page Config
st.set_page_config(page_title="AI Code Generator", layout="centered")

# Load Pretrained Models
@st.cache_resource
def load_codegen_model():
    """Load and cache the CodeGen model with a more efficient approach"""
    try:
        with st.spinner("Loading CodeGen model (this might take a minute the first time)..."):
            # Use a slightly larger model if available on your system
            tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono", legacy=False)
            model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono", 
                                                      torch_dtype=torch.float32,  # Use float16 if you have GPU support
                                                      low_cpu_mem_usage=True)
            return tokenizer, model
    except Exception as e:
        st.error(f"Error loading CodeGen model: {e}")
        return None, None

@st.cache_resource
def load_whisper_model():
    """Load and cache the Whisper model with more options"""
    try:
        with st.spinner("Loading Whisper model..."):
            # Allow selecting different model sizes based on system capacity
            model_size = "base"  # Options: "tiny", "base", "small", "medium", "large"
            return whisper.load_model(model_size)
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

tokenizer, model = load_codegen_model()
whisper_model = load_whisper_model()

# Check if models are loaded correctly
def check_models():
    """Check if models are loaded and working properly"""
    models_ok = True
    
    # Check CodeGen model
    if tokenizer is None or model is None:
        st.error("⚠️ CodeGen model failed to load. Some features may not work correctly.")
        models_ok = False
    
    # Check Whisper model
    if whisper_model is None:
        st.warning("⚠️ Whisper model failed to load. Voice input will not be available.")
        models_ok = False
    
    return models_ok

# Helper function for language examples
@lru_cache(maxsize=32)
def get_language_example(language):
    """Get a simple code example for the target language"""
    examples = {
        "Python": "def greet(name):\n    return f\"Hello, {name}!\"",
        "Java": "public class Greeter {\n    public static String greet(String name) {\n        return \"Hello, \" + name + \"!\";\n    }\n}",
        "C++": "#include <string>\n\nstd::string greet(const std::string& name) {\n    return \"Hello, \" + name + \"!\";\n}",
        "JavaScript": "function greet(name) {\n    return `Hello, ${name}!`;\n}",
        "C#": "public static class Greeter {\n    public static string Greet(string name) {\n        return $\"Hello, {name}!\";\n    }\n}",
        "Go": "package main\n\nfunc Greet(name string) string {\n    return \"Hello, \" + name + \"!\"\n}",
        "Ruby": "def greet(name)\n    \"Hello, #{name}!\"\nend",
        "Rust": "fn greet(name: &str) -> String {\n    format!(\"Hello, {}!\", name)\n}",
        "PHP": "<?php\nfunction greet($name) {\n    return \"Hello, $name!\";\n}\n?>",
        "Swift": "func greet(name: String) -> String {\n    return \"Hello, \\(name)!\"\n}",
        "TypeScript": "function greet(name: string): string {\n    return `Hello, ${name}!`;\n}"
    }
    return examples.get(language, "# No example available for this language")

# Function to generate code
def generate_code(prompt):
    """
    Generate Python code based on a natural language prompt.
    
    Args:
        prompt (str): Natural language description of the code to generate
        
    Returns:
        str: Generated Python code
    """
    # Improve the prompt to encourage executable code
    improved_prompt = f"# Python function\n{prompt}\n\ndef"
    
    inputs = tokenizer(improved_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,        # Increased max length
            temperature=0.2,       # Slightly reduced temperature for more predictable outputs
            top_p=0.95,            # Nucleus sampling for more focused outputs
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the generated code to extract only a complete function
    # Extract function definition after the prompt
    if improved_prompt in generated_code:
        generated_code = generated_code.split(improved_prompt)[-1].strip()
    
    # Ensure we start with def
    if not generated_code.startswith("def "):
        if generated_code.startswith(" "):
            generated_code = "def" + generated_code
        else:
            generated_code = "def " + generated_code
    
    # Extract only the first complete function definition
    lines = generated_code.split('\n')
    cleaned_lines = []
    in_function = False
    indentation_level = 0
    
    for i, line in enumerate(lines):
        if i == 0 or (line.strip().startswith("def ") and not in_function):
            # Start of a function
            in_function = True
            cleaned_lines = [line]
            # Check indentation of the first line after def
            if i+1 < len(lines) and lines[i+1].strip() and not lines[i+1].strip().startswith("def"):
                indentation_level = len(lines[i+1]) - len(lines[i+1].lstrip())
        elif in_function:
            # If we hit another def or test code, stop
            if line.strip().startswith("def ") or (line.strip() and not line.startswith(" " * indentation_level) and indentation_level > 0):
                # This is likely test code or a new function, stop here
                break
            # Add lines that are part of the current function
            cleaned_lines.append(line)
    
    # Join the lines back together
    clean_code = "\n".join(cleaned_lines)
    
    # Make sure the function code is properly terminated
    if not clean_code.strip().endswith(":"):
        # Find last occurrence of ":" to check if function has a body
        last_colon = clean_code.rfind(":")
        if last_colon == -1 or last_colon >= len(clean_code) - 1:
            # No function body, add a pass statement
            clean_code += ":\n    pass"
    
    return clean_code.strip()

# Function to convert code to different programming languages


# Function to execute code with test cases
def execute_code(code):
    try:
        # Create a namespace to store the function
        namespace = {}
        
        # Output buffer to capture all printed output
        output_buffer = io.StringIO()
        
        # Execute the code with output redirection
        with contextlib.redirect_stdout(output_buffer):
            # First, execute the entire code to capture any print statements
            # or code that runs at the module level
            exec(code, namespace)
            
            # Now try to find function definitions to test them
            function_name = None
            for line in code.split('\n'):
                if line.strip().startswith('def '):
                    function_name = line.strip().split('def ')[1].split('(')[0].strip()
                    break
            
            # If we found a function, test it with example inputs
            if function_name:
                function_obj = namespace.get(function_name)
                
                print(f"\n--- Testing function: {function_name} ---")
                
                # Create some example inputs based on function name/type
                test_inputs = []
                
                if "palindrome" in code.lower():
                    test_inputs = [121, 12321, 123, 1001]
                elif "fibonacci" in code.lower():
                    test_inputs = [5, 10]
                elif "prime" in code.lower():
                    test_inputs = [7, 10, 13, 20]
                elif "factorial" in code.lower():
                    test_inputs = [5, 0, 10]
                elif "sort" in code.lower():
                    test_inputs = [[5,2,9,1,5], [3,1,4]]
                elif "revers" in code.lower():
                    test_inputs = ["hello", "python", "racecar"]
                elif "anagram" in code.lower():
                    test_inputs = [("listen", "silent"), ("hello", "world")]
                else:
                    # Generic test cases
                    import inspect
                    sig = inspect.signature(function_obj)
                    param_count = len(sig.parameters)
                    
                    if param_count == 0:
                        test_inputs = [None]  # Just call the function with no args
                    elif param_count == 1:
                        test_inputs = [5, 10, "test"]  # Try both numeric and string
                    elif param_count == 2:
                        test_inputs = [(5, 10), (100, 200), ("hello", "world")]
                    else:
                        test_inputs = [(1, 2, 3)]
                
                # Execute with test inputs
                for test_input in test_inputs:
                    if test_input is None:
                        result = function_obj()
                        print(f"{function_name}() → {result}")
                    elif isinstance(test_input, tuple):
                        try:
                            result = function_obj(*test_input)
                            print(f"{function_name}{test_input} → {result}")
                        except Exception as e:
                            print(f"{function_name}{test_input} → Error: {str(e)}")
                    else:
                        try:
                            result = function_obj(test_input)
                            print(f"{function_name}({test_input}) → {result}")
                        except Exception as e:
                            print(f"{function_name}({test_input}) → Error: {str(e)}")
        
        return output_buffer.getvalue(), None
    except Exception as e:
        return None, f"❌ Error executing code: {e}"

# Transcribe audio from mic with improved accuracy
def transcribe_audio(audio_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_webm:
            tmp_webm.write(audio_bytes)
            tmp_webm.flush()

            tmp_wav = tmp_webm.name.replace(".webm", ".wav")
            ffmpeg.input(tmp_webm.name).output(tmp_wav).run(quiet=True, overwrite_output=True)

            # Use a better model and increase the temperature for more accurate transcription
            # For English specifically, use the "en" language option
            result = whisper_model.transcribe(
                tmp_wav, 
                language="en",  # Force English language
                fp16=False      # More stable on CPU
            )
            
            transcription = result["text"].strip()
            
            # Post-process the transcription to correct common coding phrases
            common_corrections = {
                "reverse a string": ["ivasomie", "reverse the string", "reverse string", "rivers a string"],
                "check if palindrome": ["check if palindrome", "check palindrome", "check if a palindrome"],
                "fibonacci sequence": ["fibonacci", "fibonacci series", "fibonacci numbers"],
                "sort an array": ["sort array", "sort the array", "sorting array"],
                "prime number": ["prime numbers", "check prime", "is prime"],
                "factorial": ["calculate factorial", "compute factorial", "find factorial"],
                "anagram": ["check anagram", "is anagram", "detect anagram"]
            }
            
            # Try to match and correct the transcription
            for correct_phrase, variations in common_corrections.items():
                for variation in variations:
                    if variation.lower() in transcription.lower():
                        return f"Write a Python function to {correct_phrase}"
            
            # Add context for programming if it seems like a coding request
            coding_keywords = ["function", "code", "write", "program", "calculate", "compute", 
                              "create", "implement", "develop", "algorithm"]
                              
            for keyword in coding_keywords:
                if keyword.lower() in transcription.lower() and "function" not in transcription.lower():
                    return f"Write a Python function to {transcription}"
            
            return transcription

    except Exception as e:
        return f"❌ Transcription failed: {e}"
def convert_code(code, target_language):
    """
    Convert Python code to the target programming language.
    
    Args:
        code (str): The Python code to convert
        target_language (str): The target programming language
    
    Returns:
        str: The converted code in the target language
    """
    # Ensure code is not empty
    if not code or not code.strip():
        return f"// Please provide Python code to convert to {target_language}"
    
    # Format a clearer and more explicit prompt for more reliable conversion
    conversion_prompt = f"""
    Convert the following Python code to {target_language}.
    
    Python code:
    ```python
    {code}
    ```
    
    {target_language} code:
    ```{target_language.lower()}
    """
    
    # Use the CodeGen model for conversion
    inputs = tokenizer(conversion_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=1024,  # Increased for longer code
            temperature=0.1,  # Lower temperature for more deterministic output
            top_p=0.95,       # Nucleus sampling for better quality
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    converted_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the converted code part
    if "```" in converted_code:
        # Find the part after our prompt and before the next triple backtick
        start_marker = f"```{target_language.lower()}"
        if start_marker in converted_code:
            code_part = converted_code.split(start_marker, 1)[1]
            if "```" in code_part:
                converted_code = code_part.split("```", 1)[0].strip()
            else:
                converted_code = code_part.strip()
        else:
            # Fallback to generic extraction if the specific marker isn't found
            parts = converted_code.split("```")
            if len(parts) >= 3:  # Find content between backticks
                converted_code = parts[2].strip()
            else:
                # Extract everything after the original prompt as a last resort
                converted_code = converted_code.split(conversion_prompt)[-1].strip()
    else:
        # If no code blocks, just take everything after the prompt
        converted_code = converted_code.split(conversion_prompt)[-1].strip()
    
    # Check if the conversion looks reasonable
    if len(converted_code) < 10 or "{" not in converted_code and target_language.lower() != "python":
        # Provide a more structured conversion template based on language
        if target_language.lower() == "java":
            return create_java_conversion(code)
        elif target_language.lower() == "c++":
            return create_cpp_conversion(code)
        elif target_language.lower() == "javascript":
            return create_js_conversion(code)
        else:
            # Default fallback
            return f"// The model couldn't generate proper {target_language} code.\n// Here's a template structure:\n\n" + converted_code
    
    return converted_code

# Helper functions for specific language conversions when the model fails
def create_java_conversion(python_code):
    """Create a structured Java conversion when the model fails"""
    # Extract function name and parameters from Python code
    import re
    func_match = re.search(r'def\s+(\w+)\s*\((.*?)\):', python_code)
    
    if not func_match:
        return """
public class Solution {
    // Could not determine function signature from Python code
    public static void main(String[] args) {
        // Implementation needed
        System.out.println("Python to Java conversion needs manual completion");
    }
}"""
    
    func_name = func_match.group(1)
    params = func_match.group(2).split(',')
    
    # Try to determine return type and param types
    return_type = "Object"  # Default
    if "return" in python_code:
        if "return True" in python_code or "return False" in python_code:
            return_type = "boolean"
        elif re.search(r'return\s+\d+', python_code):
            return_type = "int"
        elif re.search(r'return\s+"', python_code) or re.search(r"return\s+'", python_code):
            return_type = "String"
    
    # Format parameters
    java_params = []
    for i, p in enumerate(params):
        p = p.strip()
        if not p:
            continue
        
        # Determine parameter type based on context
        param_type = "Object"
        if "str" in python_code or "string" in python_code.lower():
            param_type = "String"
        elif "int" in python_code or "num" in func_name or "number" in func_name:
            param_type = "int"
        elif "list" in python_code or "[" in python_code:
            param_type = "int[]"  # Assume int array by default
        
        java_params.append(f"{param_type} {p}")
    
    java_params_str = ", ".join(java_params)
    
    # Create Java structure
    java_code = f"""
public class Solution {{
    public static {return_type} {func_name}({java_params_str}) {{
        // TODO: Convert the Python logic to Java
        // Python code was:
        /*
{python_code}
        */
        
        // Default return value
        {get_default_return(return_type)}
    }}
    
    public static void main(String[] args) {{
        // Test the function
        // Example call: {get_example_call(func_name, params, return_type)}
    }}
}}"""
    
    return java_code

def create_cpp_conversion(python_code):
    """Create a structured C++ conversion when the model fails"""
    # Extract function name and parameters
    import re
    func_match = re.search(r'def\s+(\w+)\s*\((.*?)\):', python_code)
    
    if not func_match:
        return """
#include <iostream>
#include <vector>
#include <string>

// Could not determine function signature from Python code
int main() {
    // Implementation needed
    std::cout << "Python to C++ conversion needs manual completion" << std::endl;
    return 0;
}"""
    
    func_name = func_match.group(1)
    params = func_match.group(2).split(',')
    
    # Determine return type
    return_type = "auto"  # C++11 auto for simplicity
    if "return True" in python_code or "return False" in python_code:
        return_type = "bool"
    elif re.search(r'return\s+\d+', python_code):
        return_type = "int"
    elif re.search(r'return\s+"', python_code) or re.search(r"return\s+'", python_code):
        return_type = "std::string"
    
    # Format parameters
    cpp_params = []
    for i, p in enumerate(params):
        p = p.strip()
        if not p:
            continue
        
        # Determine parameter type
        param_type = "auto"
        if "str" in python_code or "string" in python_code.lower():
            param_type = "const std::string&"
        elif "int" in python_code or "num" in func_name or "number" in func_name:
            param_type = "int"
        elif "list" in python_code or "[" in python_code:
            param_type = "const std::vector<int>&"
        
        cpp_params.append(f"{param_type} {p}")
    
    cpp_params_str = ", ".join(cpp_params)
    
    # Create C++ structure
    cpp_code = f"""
#include <iostream>
#include <vector>
#include <string>

{return_type} {func_name}({cpp_params_str}) {{
    // TODO: Convert the Python logic to C++
    // Python code was:
    /*
{python_code}
    */
    
    // Default return value
    {get_cpp_default_return(return_type)}
}}

int main() {{
    // Test the function
    // Example: {get_cpp_example_call(func_name, params, return_type)}
    return 0;
}}"""
    
    return cpp_code

def create_js_conversion(python_code):
    """Create a structured JavaScript conversion when the model fails"""
    # Extract function name and parameters
    import re
    func_match = re.search(r'def\s+(\w+)\s*\((.*?)\):', python_code)
    
    if not func_match:
        return """
/**
 * Could not determine function signature from Python code
 */
function unknownFunction() {
    // Implementation needed
    console.log("Python to JavaScript conversion needs manual completion");
}

// Test the function
// unknownFunction();
"""
    
    func_name = func_match.group(1)
    params = [p.strip() for p in func_match.group(2).split(',') if p.strip()]
    
    # Create JavaScript function
    js_code = f"""
/**
 * JavaScript implementation converted from Python
 * Python code was:
 * {python_code.split('\n')[0]}...
 */
function {func_name}({', '.join(params)}) {{
    // TODO: Convert the Python logic to JavaScript
    /*
{python_code}
    */
    
    // Default return
    return null;
}}

// Test the function
console.log("{func_name} example:", {func_name}({', '.join(['null' if not p else '"test"' if 'str' in python_code.lower() else '5' for p in params])}));
"""
    
    return js_code

def get_default_return(return_type):
    """Get default return statement based on return type"""
    if return_type == "boolean":
        return "return false;"
    elif return_type == "int":
        return "return 0;"
    elif return_type == "String":
        return "return \"\";"
    elif return_type == "int[]":
        return "return new int[0];"
    else:
        return "return null;"

def get_example_call(func_name, params, return_type):
    """Get example function call for Java"""
    param_values = []
    for p in params:
        p = p.strip()
        if not p:
            continue
            
        if "str" in p or "string" in p:
            param_values.append("\"example\"")
        elif "num" in p or "int" in p:
            param_values.append("42")
        elif "list" in p or "array" in p:
            param_values.append("new int[]{1, 2, 3}")
        else:
            param_values.append("null")
    
    params_str = ", ".join(param_values)
    
    if return_type == "void":
        return f"{func_name}({params_str});"
    else:
        return f"{return_type} result = {func_name}({params_str}); System.out.println(result);"

def get_cpp_default_return(return_type):
    """Get default return statement for C++"""
    if return_type == "bool":
        return "return false;"
    elif return_type == "int":
        return "return 0;"
    elif return_type == "std::string":
        return "return \"\";"
    elif "vector" in return_type:
        return "return {};"
    elif return_type == "auto":
        return "return nullptr;"
    else:
        return "return {};"

def get_cpp_example_call(func_name, params, return_type):
    """Get example function call for C++"""
    param_values = []
    for p in params:
        p = p.strip()
        if not p:
            continue
            
        if "str" in p or "string" in p:
            param_values.append("\"example\"")
        elif "num" in p or "int" in p:
            param_values.append("42")
        elif "list" in p or "vector" in p:
            param_values.append("{1, 2, 3}")
        else:
            param_values.append("0") # C++ doesn't have null for primitives
    
    params_str = ", ".join(param_values)
    
    if return_type == "void":
        return f"{func_name}({params_str});"
    else:
        return f"auto result = {func_name}({params_str}); std::cout << \"Result: \" << result << std::endl;"

# Helper function to get file extension for different languages
def get_file_extension(language):
    extensions = {
        "Java": "java",
        "C++": "cpp",
        "JavaScript": "js",
        "C#": "cs",
        "Go": "go",
        "Ruby": "rb",
        "Rust": "rs",
        "PHP": "php",
        "Swift": "swift",
        "TypeScript": "ts"
    }
    return extensions.get(language, "txt")

# Initialize session state for code
if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""
if "transcribed_prompt" not in st.session_state:
    st.session_state.transcribed_prompt = ""
if "converted_code" not in st.session_state:
    st.session_state.converted_code = ""
if "converted_language" not in st.session_state:
    st.session_state.converted_language = ""

# Check if models are loaded correctly
models_ok = check_models()

# UI Layout
st.title("💡 AI Code Generator & Converter 🤖")
st.write("Enter or speak your coding task. The AI will generate Python code and can convert it to other languages.")

# Only display features if models are loaded
if models_ok:
    # Mic input
    st.subheader("🎤 Speak your coding task:")
    audio_data = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording")

    if isinstance(audio_data, dict) and "bytes" in audio_data:
        with st.spinner("Transcribing..."):
            transcription = transcribe_audio(audio_data["bytes"])
            if transcription:
                st.session_state.transcribed_prompt = transcription
                st.success(f"📝 Transcription: {transcription}")
            else:
                st.warning("⚠️ No speech detected.")

    # Text prompt input
    default_prompt = "Write a Python function to check if a number is a palindrome."
    prompt = st.text_area(
        "🔹 Or type your coding task:",
        value=st.session_state.get("transcribed_prompt", default_prompt)
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⚡ Generate Code", key="generate_code_button"):
            with st.spinner("Generating code... ⏳"):
                st.session_state.generated_code = generate_code(prompt)
                st.success("✅ Code generated successfully!")

    with col2:
        if st.button("🚀 Run Code", key="run_code_button1"):
            if st.session_state.get("generated_code"):
                with st.spinner("Running code..."):
                    output, error = execute_code(st.session_state.generated_code)
                    
                    # Show the generated code
                    st.subheader("✅ Generated Code:")
                    st.code(st.session_state.generated_code, language="python")
                    
                    # Show output
                    st.subheader("🚀 Sample Output:")
                    if output:
                        st.code(output)
                    if error:
                        st.error(error)
            else:
                st.warning("⚠️ Please generate code first!")

    # Always show generated code if available
    if st.session_state.get("generated_code") and not st.button("🚀 Run Code", key="run_code_button2"):
        st.subheader("✅ Generated Code:")
        st.code(st.session_state.generated_code, language="python")

    # Add Edit and Run Options
    st.subheader("✏️ Edit the Code:")
    edited_code = st.text_area("Edit the code", value=st.session_state.get("generated_code", ""), height=300)

    if st.button("▶️ Run Edited Code", key="run_edited_code_button"):
        if edited_code:
            with st.spinner("Running edited code..."):
                output, error = execute_code(edited_code)
                
                st.subheader("🚀 Output of Edited Code:")
                if output:
                    st.code(output)
                if error:
                    st.error(error)
        else:
            st.warning("⚠️ No code to run!")

    # Download button for Python code
    if edited_code:
        st.download_button(
            label="📥 Download Python Code",
            data=edited_code,
            file_name="generated_code.py",
            mime="text/python",
            key="download_original_python_code"
        )

    # Code conversion section
    st.markdown("---")
    st.subheader("🔄 Convert Code to Another Language:")

    # Add model disclaimer
    st.caption("Note: The code conversion uses a 350M parameter model. For complex code, results may need manual adjustment.")

    # Two columns for language selection and conversion button
    col1, col2 = st.columns([3, 1])

    with col1:
        target_languages = ["Java", "C++", "JavaScript", "C#", "Go", "Ruby", "Rust", "PHP", "Swift", "TypeScript"]
        selected_language = st.selectbox("Select target language:", target_languages)

    with col2:
        convert_button = st.button("🔄 Convert Code", key="convert_code_button", 
                                help="Convert the code above to the selected language")

    # Show a progress indicator during conversion
    if convert_button:
        if edited_code:
            with st.spinner(f"Converting to {selected_language}..."):
                progress_bar = st.progress(0)
                
                # Update progress to show activity
                for i in range(5):
                    progress_bar.progress((i+1) * 20)
                    time.sleep(0.1)  # Small delay to show progress
                
                # Perform the conversion
                st.session_state.converted_code = convert_code(edited_code, selected_language)
                st.session_state.converted_language = selected_language
                
                # Complete the progress
                progress_bar.progress(100)
                
                if st.session_state.converted_code:
                    st.success(f"✅ Successfully converted to {selected_language}")
                    
                    # Show converted code with syntax highlighting
                    st.subheader(f"🖥️ {selected_language} Code:")
                    st.code(st.session_state.converted_code, language=selected_language.lower())
                    
                    # Add download button for converted code
                    file_ext = get_file_extension(selected_language)
                    st.download_button(
                        label=f"📥 Download {selected_language} Code",
                        data=st.session_state.converted_code,
                        file_name=f"converted_code.{file_ext}",
                        mime=f"text/{file_ext}",
                        key=f"download_converted_code_{selected_language}"
                    )
                    
                    # Add option to edit the converted code
                    st.subheader("✏️ Edit Converted Code:")
                    converted_edited = st.text_area(
                        "Make adjustments to the converted code if needed:",
                        value=st.session_state.converted_code,
                        height=300,
                        key=f"edit_converted_{selected_language}"
                    )
                    
                    # Update download button for edited converted code
                    if converted_edited != st.session_state.converted_code:
                        st.download_button(
                            label=f"📥 Download Edited {selected_language} Code",
                            data=converted_edited,
                            file_name=f"edited_converted_code.{file_ext}",
                            mime=f"text/{file_ext}",
                            key=f"download_edited_converted_{selected_language}"
                        )
                else:
                    st.error("❌ Conversion failed. The model couldn't generate valid code.")
        else:
            st.warning("⚠️ No code to convert! Please generate or enter some Python code first.")

    # Show previously converted code if available
    elif st.session_state.get("converted_code"):
        st.subheader(f"🖥️ Previously Converted {st.session_state.get('converted_language')} Code:")
        st.code(st.session_state.converted_code, language=st.session_state.get('converted_language').lower())
        
        # Add download button for previously converted code
        file_ext = get_file_extension(st.session_state.get('converted_language'))
        st.download_button(
            label=f"📥 Download {st.session_state.get('converted_language')} Code",
            data=st.session_state.converted_code,
            file_name=f"converted_code.{file_ext}",
            mime=f"text/{file_ext}",
            key=f"download_previous_converted_{st.session_state.get('converted_language')}"
        )

    # Show language examples section
    with st.expander("🔍 View Code Examples"):
        st.write("Reference examples of how code looks in different languages:")
        lang_tabs = st.tabs(target_languages)
        
        for i, lang in enumerate(target_languages):
            with lang_tabs[i]:
                st.code(get_language_example(lang), language=lang.lower())

else:
    st.error("⚠️ One or more models failed to load. Please check your installation and try again.")

st.markdown("---")
st.markdown("🔹 Models: Salesforce/codegen-350M-mono (Code Generation & Conversion) + Whisper (Speech-to-Text)")
st.markdown("🔹 Supported Languages: Python, Java, C++, JavaScript, C#, Go, Ruby, Rust, PHP, Swift, TypeScript")
