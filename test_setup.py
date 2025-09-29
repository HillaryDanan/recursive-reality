"""
Test the entire setup - APIs and ground truth
WITH BETTER ERROR HANDLING AND DIAGNOSTICS
"""
import asyncio
import sys
from pathlib import Path
import traceback
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.api_handlers import LLMHandler
from utils.ground_truth_generator import GroundTruthGenerator

def check_environment():
    """Check if environment is properly set up"""
    print("\n📋 ENVIRONMENT CHECK")
    print("-"*40)
    
    # Check API keys
    keys = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY')
    }
    
    for key_name, key_value in keys.items():
        if key_value:
            # Show first 8 chars of key for verification
            masked = key_value[:8] + "..." if len(key_value) > 8 else "NOT SET"
            print(f"  {key_name}: {masked}")
        else:
            print(f"  {key_name}: ❌ NOT SET")
    
    return any(keys.values())

async def test_everything():
    """Test all components with better error handling"""
    print("\n" + "="*60)
    print("RECURSIVE REALITY SETUP TEST")
    print("="*60)
    
    # Check environment first
    if not check_environment():
        print("\n⚠️  No API keys found!")
        print("Please add at least one API key to your .env file:")
        print("  OPENAI_API_KEY=sk-...")
        print("  ANTHROPIC_API_KEY=sk-ant-...")
        print("  GOOGLE_API_KEY=...")
        return False
    
    # Test APIs
    print("\n🔌 TESTING API CONNECTIONS")
    print("-"*40)
    
    try:
        handler = LLMHandler()
        api_results = await handler.test_all_apis()
    except Exception as e:
        print(f"\n❌ Failed to initialize API handler: {e}")
        print("\nDEBUG INFO:")
        print(traceback.format_exc())
        return False
    
    working_apis = [api for api, status in api_results.items() if status]
    
    if not working_apis:
        print("\n⚠️  No APIs are working. Check:")
        print("  1. Your API keys are valid")
        print("  2. You have credits/quota remaining")
        print("  3. Your network connection")
        return False
    
    print(f"\n✅ Working APIs: {', '.join(working_apis)}")
    
    # Check ground truth
    print("\n📊 CHECKING GROUND TRUTH DATA")
    print("-"*40)
    
    gt_path = Path("data/ground_truth/scientific_facts.csv")
    if gt_path.exists():
        import pandas as pd
        facts_df = pd.read_csv(gt_path)
        print(f"✅ Ground truth loaded: {len(facts_df)} facts")
        print(f"   Categories: {facts_df['category'].nunique()}")
        print(f"   Complexities: {facts_df['complexity'].unique().tolist()}")
    else:
        print("⚠️  Ground truth not found, generating...")
        generator = GroundTruthGenerator()
        facts_df = generator.generate_scientific_facts()
        generator.save_ground_truth(facts_df)
    
    # Test a simple degradation
    print("\n🧪 TESTING DEGRADATION PIPELINE")
    print("-"*40)
    
    test_fact = facts_df.iloc[0]['fact']
    print(f"Original fact: {test_fact[:100]}...")
    
    for api in working_apis[:1]:  # Test first working API
        model_map = {
            'openai': 'gpt-4',
            'anthropic': 'claude-3-sonnet',
            'google': 'gemini-pro'
        }
        
        try:
            model = model_map[api]
            print(f"\nTesting {model}...")
            response = await handler.query(
                model,
                f"Restate this scientific fact in your own words: {test_fact}"
            )
            print(f"Response: {response[:150]}...")
            print(f"✅ {model} working correctly")
        except Exception as e:
            print(f"❌ {model} failed: {e}")
    
    # Save test results
    if handler.call_history:
        handler.save_history("logs/test_history.json")
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE - READY FOR EXPERIMENTS!")
    print(f"   Working APIs: {len(working_apis)}")
    print(f"   Ground truth facts: {len(facts_df)}")
    print("="*60)
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(test_everything())
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print(traceback.format_exc())
        sys.exit(1)
