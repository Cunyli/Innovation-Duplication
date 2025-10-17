"""
Test Azure OpenAI connection with real API calls
"""
import sys
from pathlib import Path

# Add parent directory to path to import config module
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

def test_chat_model():
    """Test GPT chat model connection"""
    print("\n🔍 Testing GPT-4o-mini chat model...")
    try:
        config = load_config()
        model_config = config['gpt-4o-mini']
        
        llm = AzureChatOpenAI(
            model=model_config['deployment'],
            api_key=model_config['api_key'],
            azure_endpoint=model_config['api_base'],
            api_version=model_config['api_version']
        )
        
        # Test with a simple prompt
        response = llm.invoke("Say 'Hello, I am working!' in one sentence.")
        print(f"✅ Chat model response: {response.content}")
        return True
    except Exception as e:
        print(f"❌ Chat model error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_model():
    """Test embedding model connection"""
    print("\n🔍 Testing text-embedding-3-small model...")
    try:
        config = load_config()
        model_config = config['gpt-4o-mini']
        
        embedding_model = AzureOpenAIEmbeddings(
            model=model_config['emb_deployment'],
            api_key=model_config['api_key'],
            azure_endpoint=model_config['api_base'],
            api_version=model_config.get('emb_api_version', model_config['api_version'])
        )
        
        # Test with a simple text
        test_text = "This is a test sentence."
        embedding = embedding_model.embed_query(test_text)
        print(f"✅ Embedding model working! Vector dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        return True
    except Exception as e:
        print(f"❌ Embedding model error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Azure OpenAI Connection Test")
    print("=" * 60)
    
    # Test chat model
    chat_success = test_chat_model()
    
    # Test embedding model
    embedding_success = test_embedding_model()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    print(f"{'✅' if chat_success else '❌'} Chat Model (GPT-4o-mini)")
    print(f"{'✅' if embedding_success else '❌'} Embedding Model (text-embedding-3-small)")
    
    if chat_success and embedding_success:
        print("\n🎉 All tests passed! Your Azure OpenAI configuration is working!")
    else:
        print("\n⚠️ Some tests failed. Please check your configuration and API keys.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
