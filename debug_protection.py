#!/usr/bin/env python3
"""
Simple debugging script for Galileo Protect integration.
This script helps test the protection functionality step by step.
"""

import os
import sys
from dotenv import load_dotenv

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

def test_protection_chain():
    """Test the protection chain creation and basic functionality."""
    
    print("🔧 Testing Protection Chain Creation")
    print("=" * 40)
    
    try:
        # Import the function
        from app import create_protected_chain
        
        print("✅ Successfully imported create_protected_chain")
        
        # Test chain creation
        print("\n📦 Creating protected chain...")
        protected_chain, galileo_callback = create_protected_chain(
            model="gpt-4o-mini",
            timeout=5
        )
        
        print("✅ Protected chain created successfully")
        print(f"   Chain type: {type(protected_chain)}")
        print(f"   Callback type: {type(galileo_callback)}")
        
        # Test with a simple query
        print("\n🧪 Testing with simple query...")
        test_query = "Hello, how are you?"
        
        try:
            response = protected_chain.invoke(
                {"input": test_query},
                config={"callbacks": [galileo_callback]}
            )
            
            print("✅ Chain invocation successful")
            print(f"   Response type: {type(response)}")
            print(f"   Response content: {response}")
            
            if hasattr(response, 'content'):
                print(f"   Response.content: {response.content}")
            
        except Exception as e:
            print(f"❌ Chain invocation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

def test_environment():
    """Test if required environment variables are set."""
    
    print("🔧 Testing Environment Variables")
    print("=" * 40)
    
    required_vars = [
        "OPENAI_API_KEY",
        "GALILEO_API_KEY", 
        "GALILEO_PROJECT_NAME"
    ]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {'*' * 8}{value[-4:] if len(value) > 4 else '***'}")
        else:
            print(f"❌ {var}: Not set")

def test_imports():
    """Test if all required imports work."""
    
    print("🔧 Testing Imports")
    print("=" * 40)
    
    try:
        from galileo.handlers.langchain import GalileoCallback
        print("✅ GalileoCallback imported")
    except Exception as e:
        print(f"❌ GalileoCallback import failed: {str(e)}")
    
    try:
        from galileo.handlers.langchain.tool import ProtectTool, ProtectParser
        print("✅ ProtectTool, ProtectParser imported")
    except Exception as e:
        print(f"❌ ProtectTool, ProtectParser import failed: {str(e)}")
    
    try:
        from galileo_core.schemas.protect.ruleset import Ruleset
        print("✅ Ruleset imported")
    except Exception as e:
        print(f"❌ Ruleset import failed: {str(e)}")
    
    try:
        from langchain_openai import ChatOpenAI
        print("✅ ChatOpenAI imported")
    except Exception as e:
        print(f"❌ ChatOpenAI import failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Galileo Protect Debug Script")
    print("=" * 50)
    
    # Test environment
    test_environment()
    print()
    
    # Test imports
    test_imports()
    print()
    
    # Test protection chain
    test_protection_chain()
    print()
    
    print("🏁 Debug script complete!") 