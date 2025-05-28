import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_secrets_to_env():
    """
    Convert .streamlit/secrets.toml to .env file.
    """
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    env_path = ".env"
    
    logger.info(f"Reading secrets from {secrets_path}")
    
    try:
        # Read the TOML file line by line and extract key-value pairs
        secrets = {}
        with open(secrets_path, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
                
            # Split by first equals sign
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                    
                secrets[key] = value
        
        # Create .env content
        env_lines = []
        
        # Add environment variables for Galileo
        env_lines.append("# Galileo Configuration")
        env_lines.append(f"GALILEO_API_KEY={secrets.get('galileo_api_key', '')}")
        env_lines.append(f"GALILEO_PROJECT={secrets.get('galileo_project', '')}")
        env_lines.append(f"GALILEO_LOG_STREAM={secrets.get('galileo_log_stream', '')}")
        env_lines.append(f"GALILEO_CONSOLE_URL={secrets.get('galileo_console_url', '')}")
        
        # Add OpenAI API key
        env_lines.append("\n# OpenAI Configuration")
        env_lines.append(f"OPENAI_API_KEY={secrets.get('openai_api_key', '')}")
        
        # Add Pinecone Configuration
        env_lines.append("\n# Pinecone Configuration")
        env_lines.append(f"PINECONE_API_KEY={secrets.get('pinecone_api_key', '')}")
        env_lines.append(f"PINECONE_INDEX_NAME={secrets.get('pinecone_index_name', '')}")
        env_lines.append(f"PINECONE_NAMESPACE={secrets.get('pinecone_namespace', '')}")
        
        # Add any other keys that might be useful
        env_lines.append("\n# Other Configuration")
        for key, value in secrets.items():
            # Skip keys we've already added
            if key in ['galileo_api_key', 'galileo_project', 'galileo_log_stream', 'galileo_console_url',
                       'openai_api_key', 'pinecone_api_key', 'pinecone_index_name', 'pinecone_namespace']:
                continue
                
            # Add the key
            env_lines.append(f"{key.upper()}={value}")
        
        # Write the .env file
        with open(env_path, "w") as f:
            f.write("\n".join(env_lines))
        
        logger.info(f"Successfully generated {env_path} file from secrets")
        logger.info(f"Found {len(secrets)} configuration items")
        
    except FileNotFoundError:
        logger.error(f"Could not find {secrets_path}")
    except Exception as e:
        logger.error(f"Error converting secrets to .env: {e}")

if __name__ == "__main__":
    convert_secrets_to_env() 