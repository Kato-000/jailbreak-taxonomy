"""
api_model_wrapper.py
APIベースのモデル（OpenAI、Anthropic、Hugging Face Inference）のラッパークラス
"""
import os
from typing import Optional, Dict, Any
import time


class APIModelWrapper:
    """APIモデルのラッパークラス"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        初期化
        
        Args:
            model_name: モデル名（例: gpt-3.5-turbo, claude-sonnet-4-20250514）
            api_key: APIキー（Noneの場合は環境変数から取得）
        """
        self.model_name = model_name
        self.api_type = self._detect_api_type(model_name)
        
        # APIクライアントの初期化
        if self.api_type == "openai":
            self._init_openai(api_key)
        elif self.api_type == "anthropic":
            self._init_anthropic(api_key)
        elif self.api_type == "huggingface":
            self._init_huggingface(api_key)
        else:
            raise ValueError(f"サポートされていないモデル: {model_name}")
    
    def _detect_api_type(self, model_name: str) -> str:
        """モデル名からAPI種別を判定"""
        model_lower = model_name.lower()
        
        if model_lower.startswith("gpt-") or "gpt" in model_lower:
            return "openai"
        elif model_lower.startswith("claude-") or "claude" in model_lower:
            return "anthropic"
        elif "/" in model_name:  # Hugging Face形式 (org/model)
            return "huggingface"
        else:
            return "unknown"
    
    def _init_openai(self, api_key: Optional[str] = None):
        """OpenAI APIの初期化"""
        try:
            import openai
            
            # APIキーの設定
            if api_key:
                openai.api_key = api_key
            else:
                openai.api_key = os.getenv("OPENAI_API_KEY")
            
            if not openai.api_key:
                raise ValueError("OpenAI APIキーが設定されていません")
            
            self.client = openai
            print(f"[INFO] OpenAI API初期化完了: {self.model_name}")
            
        except ImportError:
            raise ImportError("openaiパッケージがインストールされていません。'pip install openai'を実行してください。")
    
    def _init_anthropic(self, api_key: Optional[str] = None):
        """Anthropic APIの初期化"""
        try:
            import anthropic
            
            # APIキーの設定
            if api_key:
                key = api_key
            else:
                key = os.getenv("ANTHROPIC_API_KEY")
            
            if not key:
                raise ValueError("Anthropic APIキーが設定されていません")
            
            self.client = anthropic.Anthropic(api_key=key)
            print(f"[INFO] Anthropic API初期化完了: {self.model_name}")
            
        except ImportError:
            raise ImportError("anthropicパッケージがインストールされていません。'pip install anthropic'を実行してください。")
    
    def _init_huggingface(self, api_key: Optional[str] = None):
        """Hugging Face Inference APIの初期化"""
        try:
            from huggingface_hub import InferenceClient
            
            # APIキーの設定
            if api_key:
                token = api_key
            else:
                token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
            
            self.client = InferenceClient(token=token)
            print(f"[INFO] Hugging Face Inference API初期化完了: {self.model_name}")
            
        except ImportError:
            raise ImportError("huggingface_hubパッケージがインストールされていません。'pip install huggingface_hub'を実行してください。")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        テキスト生成
        
        Args:
            prompt: 入力プロンプト
            max_tokens: 最大生成トークン数
            temperature: サンプリング温度
            **kwargs: その他のパラメータ
        
        Returns:
            生成結果の辞書
        """
        if self.api_type == "openai":
            return self._generate_openai(prompt, max_tokens, temperature, **kwargs)
        elif self.api_type == "anthropic":
            return self._generate_anthropic(prompt, max_tokens, temperature, **kwargs)
        elif self.api_type == "huggingface":
            return self._generate_huggingface(prompt, max_tokens, temperature, **kwargs)
    
    def _generate_openai(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict[str, Any]:
        """OpenAI APIでテキスト生成"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            generated_text = response.choices[0].message.content
            
            return {
                "text": generated_text,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "raw_response": response
            }
            
        except Exception as e:
            print(f"[ERROR] OpenAI API呼び出しエラー: {e}")
            return {"text": "", "error": str(e)}
    
    def _generate_anthropic(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Anthropic APIでテキスト生成"""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            generated_text = response.content[0].text
            
            return {
                "text": generated_text,
                "stop_reason": response.stop_reason,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                "raw_response": response
            }
            
        except Exception as e:
            print(f"[ERROR] Anthropic API呼び出しエラー: {e}")
            return {"text": "", "error": str(e)}
    
    def _generate_huggingface(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Hugging Face Inference APIでテキスト生成"""
        try:
            response = self.client.text_generation(
                prompt,
                model=self.model_name,
                max_new_tokens=max_tokens,
                temperature=temperature,
                return_full_text=False,
                **kwargs
            )
            
            return {
                "text": response,
                "raw_response": response
            }
            
        except Exception as e:
            error_str = str(e)
            
            # Gated modelのエラーを特別に処理
            if "403" in error_str or "gated" in error_str.lower() or "authorization" in error_str.lower():
                print(f"[ERROR] 制限付きモデルへのアクセスエラー: {self.model_name}")
                print(f"[INFO] Hugging Faceでアクセス申請が必要です:")
                print(f"       1. https://huggingface.co/{self.model_name} にアクセス")
                print(f"       2. 'Request access' をクリック")
                print(f"       3. 承認後、トークンでログイン: huggingface-cli login")
                return {
                    "text": "",
                    "error": f"Gated model access denied: {self.model_name}. Please request access on Hugging Face."
                }
            
            print(f"[ERROR] Hugging Face API呼び出しエラー: {e}")
            return {"text": "", "error": str(e)}
    
    def get_model_info(self) -> Dict[str, str]:
        """モデル情報を取得"""
        return {
            "model_name": self.model_name,
            "api_type": self.api_type,
            "model_type": "API Model",
            "provider": self.api_type.upper()
        }


def is_api_model(model_name: str) -> bool:
    """
    モデル名がAPIモデルかどうかを判定
    
    Args:
        model_name: モデル名
    
    Returns:
        APIモデルの場合True
    """
    model_lower = model_name.lower()
    
    # OpenAIモデル
    openai_patterns = ["gpt-3.5", "gpt-4", "gpt-3", "gpt-turbo"]
    if any(pattern in model_lower for pattern in openai_patterns):
        return True
    
    # Anthropicモデル
    if model_lower.startswith("claude-"):
        return True
    
    # ローカルフォルダやHugging Faceのモデルパスでない場合
    if "/" not in model_name and not os.path.exists(model_name):
        # 明示的なAPIモデル名のパターン
        if model_lower in ["gpt3.5", "gpt4", "claude"]:
            return True
    
    return False