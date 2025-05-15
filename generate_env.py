#!/usr/bin/env python
"""
Script to generate .env file from .streamlit/secrets.toml
"""
import re
import os

def convert_toml_to_env(input_path, output_path):
    """Convert TOML format to ENV format."""
    with open(input_path, 'r') as f:
        toml_content = f.read()
    
    # Process the content line by line
    env_lines = []
    for line in toml_content.split('\n'):
        # Skip comments and empty lines
        if line.strip().startswith('#') or not line.strip():
            env_lines.append(line)
            continue
        
        # Match key-value pairs
        match = re.match(r'([a-zA-Z_]+)\s*=\s*"([^"]*)"', line)
        if match:
            key, value = match.groups()
            # Convert to uppercase ENV format
            env_key = key.upper()
            env_lines.append(f'{env_key}="{value}"')
    
    # Write to .env file
    with open(output_path, 'w') as f:
        f.write('\n'.join(env_lines))
    
    print(f"Generated {output_path} from {input_path}")

if __name__ == "__main__":
    input_path = '.streamlit/secrets.toml'
    output_path = '.env'
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found!")
        exit(1)
    
    convert_toml_to_env(input_path, output_path) 