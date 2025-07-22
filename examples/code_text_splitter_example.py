"""
Example script demonstrating how to use the CodeTextSplitter.
This script shows how to split code using language-specific separators.
"""

from easy_rag.utils.text_splitters import get_language_enum_values, get_language_separators, split_code_example
from easy_rag.utils.code_text_splitter import CodeTextSplitter

def main():
    # Example Python code
    python_code = """
def hello_world():
    print("Hello, World!")

def calculate_sum(a, b):
    return a + b

# Call the functions
hello_world()
result = calculate_sum(5, 10)
print(f"The sum is: {result}")
"""

    # Example JavaScript code
    javascript_code = """
function helloWorld() {
    console.log("Hello, World!");
}

function calculateSum(a, b) {
    return a + b;
}

// Call the functions
helloWorld();
const result = calculateSum(5, 10);
console.log(`The sum is: ${result}`);
"""

    # Get all supported languages
    print("Supported languages:")
    code_splitter = CodeTextSplitter()
    supported_languages = code_splitter.get_supported_languages()
    for lang in supported_languages:
        print(f"- {lang}")
    print()

    # Get separators for Python
    print("Separators for Python:")
    python_separators = code_splitter.get_separators_for_language("PYTHON")
    print(python_separators)
    print()

    # Split Python code
    print("Splitting Python code:")
    python_chunks = code_splitter.split_text(
        python_code, chunk_size=100, chunk_overlap=0, language="PYTHON"
    )
    for i, chunk in enumerate(python_chunks):
        print(f"Chunk {i+1}:")
        print(chunk)
        print("-" * 40)
    print()

    # Split JavaScript code
    print("Splitting JavaScript code:")
    javascript_chunks = code_splitter.split_text(
        javascript_code, chunk_size=100, chunk_overlap=0, language="JAVASCRIPT"
    )
    for i, chunk in enumerate(javascript_chunks):
        print(f"Chunk {i+1}:")
        print(chunk)
        print("-" * 40)
    print()

    # Alternative way using the split_code_example function
    print("Using split_code_example function:")
    python_chunks_alt = split_code_example(
        python_code, language_value="PYTHON", chunk_size=100, chunk_overlap=0
    )
    for i, chunk in enumerate(python_chunks_alt):
        print(f"Chunk {i+1}:")
        print(chunk)
        print("-" * 40)

if __name__ == "__main__":
    main()