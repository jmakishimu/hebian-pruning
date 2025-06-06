# debug_env.py
import sys
import os
import transformers

print("--- Python Environment Diagnostic ---")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print("-" * 30)

print("Package Information:")
print(f"Transformers Version: {transformers.__version__}")
print(f"Transformers Library Path: {transformers.__file__}")
print("-" * 30)

print("System Path (where Python looks for imports):")
# Print each path on a new line for clarity
for p in sys.path:
    print(p)
print("-" * 30)

print("PYTHONPATH Environment Variable:")
# Check if PYTHONPATH is set and print it
python_path_var = os.environ.get('PYTHONPATH')
if python_path_var:
    print(python_path_var)
else:
    print("PYTHONPATH is not set.")
print("-" * 30)