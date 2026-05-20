#!/usr/bin/env python3
"""
test_api_models.py
各LLM APIの接続テストと利用可能なモデルの確認
"""
import os
import sys

def test_claude_api():
    """Claude APIのテスト"""
    print("\n" + "="*60)
    print("Claude API テスト")
    print("="*60)
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY が設定されていません")
        return False
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        # 試すモデルのリスト
        models = [
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-haiku-20240307"
        ]
        
        successful_model = None
        for model in models:
            try:
                print(f"\n試行中: {model}")
                message = client.messages.create(
                    model=model,
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Hi"}]
                )
                print(f"✅ {model} - 利用可能")
                if not successful_model:
                    successful_model = model
            except Exception as e:
                if "not_found" in str(e).lower():
                    print(f"❌ {model} - 見つかりません")
                else:
                    print(f"⚠️  {model} - エラー: {str(e)[:50]}...")
        
        if successful_model:
            print(f"\n推奨モデル: {successful_model}")
            return True
        else:
            print("\n❌ 利用可能なモデルが見つかりませんでした")
            return False
            
    except ImportError:
        print("❌ anthropic パッケージがインストールされていません")
        print("   pip install anthropic")
        return False
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False


def test_openai_api():
    """OpenAI APIのテスト"""
    print("\n" + "="*60)
    print("OpenAI API テスト")
    print("="*60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY が設定されていません")
        return False
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        # 試すモデルのリスト
        models = [
            "gpt-4-turbo-preview",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo"
        ]
        
        successful_model = None
        for model in models:
            try:
                print(f"\n試行中: {model}")
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=10
                )
                print(f"✅ {model} - 利用可能")
                if not successful_model:
                    successful_model = model
            except Exception as e:
                if "does not exist" in str(e) or "not found" in str(e).lower():
                    print(f"❌ {model} - 見つかりません")
                else:
                    print(f"⚠️  {model} - エラー: {str(e)[:50]}...")
        
        if successful_model:
            print(f"\n推奨モデル: {successful_model}")
            return True
        else:
            print("\n❌ 利用可能なモデルが見つかりませんでした")
            return False
            
    except ImportError:
        print("❌ openai パッケージがインストールされていません")
        print("   pip install openai")
        return False
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False


def test_llama_api():
    """Hugging Face Llama APIのテスト"""
    print("\n" + "="*60)
    print("Hugging Face Llama API テスト")
    print("="*60)
    
    api_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
    if not api_key:
        print("❌ HUGGINGFACE_API_KEY または HF_TOKEN が設定されていません")
        return False
    
    try:
        from huggingface_hub import InferenceClient
        
        client = InferenceClient(token=api_key)
        
        # 試すモデルのリスト
        models = [
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3-8B-Instruct"
        ]
        
        successful_model = None
        for model_name in models:
            try:
                print(f"\n試行中: {model_name}")
                response = client.chat_completion(
                    model=model_name,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=10
                )
                print(f"✅ {model_name} - 利用可能")
                if not successful_model:
                    successful_model = model_name
            except Exception as e:
                error_str = str(e).lower()
                if "not found" in error_str or "does not exist" in error_str or "not available" in error_str:
                    print(f"❌ {model_name} - 見つかりません")
                else:
                    print(f"⚠️  {model_name} - エラー: {str(e)[:50]}...")
        
        if successful_model:
            print(f"\n推奨モデル: {successful_model}")
            print("\n💡 Llama モデルへのアクセス:")
            print("  1. https://huggingface.co/ でアカウント作成")
            print("  2. Settings → Access Tokens でトークン生成")
            print("  3. Llama モデルページで利用規約に同意が必要な場合があります")
            return True
        else:
            print("\n❌ 利用可能なモデルが見つかりませんでした")
            print("\n💡 Hugging Face での Llama アクセス:")
            print("  1. https://huggingface.co/meta-llama にアクセス")
            print("  2. 各モデルページで 'Request Access' をクリック")
            print("  3. 承認後、APIトークンで利用可能")
            return False
            
    except ImportError:
        print("❌ huggingface_hub パッケージがインストールされていません")
        print("   pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False


def main():
    """メイン処理"""
    print("\n" + "="*60)
    print("LLM API 接続テスト")
    print("="*60)
    print("\n環境変数を確認中...")
    
    # 環境変数の確認
    env_vars = {
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "HUGGINGFACE_API_KEY/HF_TOKEN": os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
    }
    
    for var, value in env_vars.items():
        if value:
            print(f"✓ {var}: {'*' * 10}{value[-4:]}")
        else:
            print(f"✗ {var}: 未設定")
    
    # 各APIのテスト
    results = {
        "Claude": test_claude_api(),
        "OpenAI": test_openai_api(),
        "Llama": test_llama_api()
    }
    
    # サマリー
    print("\n" + "="*60)
    print("テスト結果サマリー")
    print("="*60)
    
    for api_name, success in results.items():
        status = "✅ 利用可能" if success else "❌ 利用不可"
        print(f"{api_name:15s}: {status}")
    
    available_count = sum(results.values())
    print(f"\n利用可能なAPI: {available_count}/3")
    
    if available_count == 0:
        print("\n⚠️  警告: 利用可能なAPIがありません")
        print("少なくとも1つのAPIキーを設定してください")
        sys.exit(1)
    elif available_count < 3:
        print("\n⚠️  注意: 一部のAPIのみ利用可能です")
        print("最高の精度を得るには、3つすべてのAPIを設定することを推奨します")
    else:
        print("\n✅ すべてのAPIが正常に動作しています！")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
