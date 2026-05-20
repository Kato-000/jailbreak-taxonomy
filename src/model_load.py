"""
model_load.py
言語モデルとトークナイザーの読み込みを管理するクラス
ローカルモデルとAPIモデルの両方をサポート
"""
import torch
import os
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# APIモデルラッパーのインポート
try:
    from .api_model_wrapper import APIModelWrapper, is_api_model
    API_SUPPORT = True
except ImportError:
    try:
        # 絶対インポートも試す（スクリプトとして実行された場合）
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from api_model_wrapper import APIModelWrapper, is_api_model
        API_SUPPORT = True
    except ImportError:
        print("[WARNING] api_model_wrapper.py not found - API models not supported")
        API_SUPPORT = False
        APIModelWrapper = None
        
        def is_api_model(model_name: str) -> bool:
            return False


class ModelLoader:
    """
    言語モデルとトークナイザーの読み込みを管理するクラス
    ローカルモデルとAPIモデルの両方をサポート
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        api_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        claude_key: Optional[str] = None,
        huggingface_key: Optional[str] = None,
        use_api: Optional[str] = None
    ):
        """
        モデルとトークナイザーの読み込み
        
        Args:
            model_path: モデルのパスまたはHugging Face model IDまたはAPI model名
            device: 使用デバイス ('auto', 'cuda', 'cpu') - ローカルモデルのみ
            api_key: APIキー（汎用）
            openai_key: OpenAI APIキー
            claude_key: Claude APIキー
            huggingface_key: Hugging Face APIキー
            use_api: 強制的にAPIモードを使用 ('huggingface', 'openai', 'anthropic', 'auto')
        """
        self.model_path = model_path
        
        # APIモデルかどうかの判定
        print(f"[DEBUG] モデルパス: {model_path}")
        print(f"[DEBUG] API_SUPPORT: {API_SUPPORT}")
        print(f"[DEBUG] use_api指定: {use_api}")
        
        self.is_api_model = False
        
        # use_apiが指定されている場合は強制的にAPIモード
        if use_api:
            if not API_SUPPORT:
                raise RuntimeError(
                    f"--use-api が指定されましたが、APIサポートが利用できません。\n"
                    f"必要なパッケージをインストールしてください:\n"
                    f"  pip install openai anthropic huggingface_hub"
                )
            self.is_api_model = True
            print(f"[INFO] --use-api={use_api} により、強制的にAPIモードを使用します")
        elif API_SUPPORT:
            self.is_api_model = is_api_model(model_path)
            print(f"[DEBUG] is_api_model判定結果: {self.is_api_model}")
        else:
            # APIサポートが無い場合でも、明らかなAPIモデル名の場合は判定
            model_lower = model_path.lower()
            if any(pattern in model_lower for pattern in ["gpt-3", "gpt-4", "gpt3", "gpt4"]):
                self.is_api_model = True
                print(f"[DEBUG] OpenAIモデルと判定（APIサポート無し）")
            elif "claude" in model_lower:
                self.is_api_model = True
                print(f"[DEBUG] Claudeモデルと判定（APIサポート無し）")
        
        # APIモデルの場合
        if self.is_api_model:
            if not API_SUPPORT:
                raise RuntimeError(
                    f"APIモデル '{model_path}' が指定されましたが、api_model_wrapper.pyが利用できません。\n"
                    f"必要なパッケージをインストールしてください:\n"
                    f"  pip install openai anthropic huggingface_hub"
                )
            
            print(f"[INFO] APIモデルを使用: {model_path}")
            
            # API種別に応じたキーを選択
            model_lower = model_path.lower()
            
            # use_apiが指定されている場合はそれを優先
            if use_api == "openai" or (use_api == "auto" and "gpt" in model_lower):
                selected_key = openai_key or api_key or os.getenv("OPENAI_API_KEY")
                api_type = "openai"
            elif use_api == "anthropic" or (use_api == "auto" and "claude" in model_lower):
                selected_key = claude_key or api_key or os.getenv("ANTHROPIC_API_KEY")
                api_type = "anthropic"
            elif use_api == "huggingface" or use_api == "auto" or "/" in model_path:
                # Hugging Face Inference API
                selected_key = huggingface_key or api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
                api_type = "huggingface"
            elif "gpt" in model_lower:
                selected_key = openai_key or api_key or os.getenv("OPENAI_API_KEY")
                api_type = "openai"
            elif "claude" in model_lower:
                selected_key = claude_key or api_key or os.getenv("ANTHROPIC_API_KEY")
                api_type = "anthropic"
            else:
                # デフォルトはHugging Face
                selected_key = huggingface_key or api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
                api_type = "huggingface"
            
            print(f"[INFO] API種別: {api_type}")
            
            if not selected_key:
                raise RuntimeError(
                    f"APIキーが設定されていません。\n"
                    f"\n"
                    f"使用するAPI: {api_type}\n"
                    f"\n"
                    f"APIキーを設定してください:\n"
                    f"  export OPENAI_API_KEY='sk-...' (OpenAI用)\n"
                    f"  export ANTHROPIC_API_KEY='sk-ant-...' (Anthropic用)\n"
                    f"  export HUGGINGFACE_API_KEY='hf_...' (Hugging Face用)\n"
                    f"\n"
                    f"または、コマンドライン引数で指定:\n"
                    f"  --openai-key 'sk-...'\n"
                    f"  --claude-key 'sk-ant-...'\n"
                    f"  --huggingface-key 'hf_...'\n"
                )
            
            self.model = APIModelWrapper(model_path, api_key=selected_key)
            self.tokenizer = None  # APIモデルはトークナイザー不要
            self.device = None  # APIモデルはデバイス指定不要
            
            print(f"[SUCCESS] APIモデル初期化完了")
            return
        
        # ローカルモデルの場合（元のコード）
        # デバイスの設定
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"[INFO] デバイス: {self.device}")
        print(f"[INFO] モデルを読み込み中: {model_path}")
        
        # ローカルパスかどうかを判定
        is_local_path = os.path.exists(model_path) or model_path.startswith('./') or model_path.startswith('../') or os.path.isabs(model_path)
        
        # ローカルパスの場合、存在確認
        if is_local_path and not os.path.exists(model_path):
            raise RuntimeError(
                f"ローカルモデルパスが見つかりません: {model_path}\n"
                f"\n"
                f"解決方法:\n"
                f"1. パスが正しいか確認してください\n"
                f"2. 絶対パスまたは相対パスで指定してください\n"
                f"3. モデルディレクトリには以下のファイルが必要です:\n"
                f"   - config.json\n"
                f"   - pytorch_model.bin (または model.safetensors)\n"
                f"   - tokenizer.json (または tokenizer_config.json)\n"
                f"\n"
                f"Hugging Faceのモデルを使用する場合:\n"
                f"  --model meta-llama/Llama-2-7b-hf\n"
                f"\n"
                f"APIモデルを使用する場合:\n"
                f"  --model gpt-3.5-turbo\n"
                f"  --model claude-sonnet-4-20250514"
            )
        
        # トークナイザーの読み込み
        try:
            # ローカルパスの場合は local_files_only=True を使用
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=is_local_path
            )
            
            # pad_tokenの設定（存在しない場合）
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"[SUCCESS] トークナイザー読み込み完了")
            
        except Exception as e:
            # Gated repositoryのエラーを特別に処理
            error_str = str(e)
            if "gated repo" in error_str.lower() or "403" in error_str:
                raise RuntimeError(
                    f"制限付きモデルへのアクセスエラー: {model_path}\n"
                    f"\n"
                    f"このモデルは「Gated Repository」（制限付きリポジトリ）です。\n"
                    f"アクセスするには、Hugging Faceでの承認とトークンが必要です。\n"
                    f"\n"
                    f"🔐 解決方法:\n"
                    f"\n"
                    f"【方法1】Hugging Faceでアクセス申請（時間がかかる）:\n"
                    f"  1. https://huggingface.co/{model_path} にアクセス\n"
                    f"  2. 'Request access' ボタンをクリックして申請\n"
                    f"  3. 承認を待つ（数分〜数時間）\n"
                    f"  4. トークンを取得: https://huggingface.co/settings/tokens\n"
                    f"  5. ログイン:\n"
                    f"     huggingface-cli login\n"
                    f"     または\n"
                    f"     export HUGGINGFACE_API_KEY='hf_...'\n"
                    f"\n"
                    f"【方法2】制限なしの別のモデルを使用（すぐに使える）:\n"
                    f"  # Mistral 7B（制限なし、軽量）\n"
                    f"  python main.py --model mistralai/Mistral-7B-Instruct-v0.2 \\\n"
                    f"    --dataset ./Dataset/AttackDataset/prompt.json\n"
                    f"\n"
                    f"  # Phi-3（制限なし、非常に軽量）\n"
                    f"  python main.py --model microsoft/Phi-3-mini-4k-instruct \\\n"
                    f"    --dataset ./Dataset/AttackDataset/prompt.json\n"
                    f"\n"
                    f"【方法3】APIモデルを使用（推奨・最も簡単）:\n"
                    f"  export OPENAI_API_KEY='sk-...'\n"
                    f"  python main.py --model gpt-3.5-turbo \\\n"
                    f"    --dataset ./Dataset/AttackDataset/prompt.json\n"
                    f"\n"
                    f"元のエラー: {error_str[:200]}..."
                )
            
            # より詳細なエラーメッセージ
            error_msg = f"トークナイザーの読み込みに失敗: {e}\n\n"
            
            if is_local_path:
                error_msg += (
                    f"ローカルパス '{model_path}' からの読み込みに失敗しました。\n"
                    f"\n"
                    f"考えられる原因:\n"
                    f"1. モデルファイルが不完全または破損している\n"
                    f"2. 必要なファイル（tokenizer.json、config.json）が不足している\n"
                    f"3. モデル形式が Hugging Face Transformers と互換性がない\n"
                    f"\n"
                    f"解決方法:\n"
                    f"- Hugging Face Hub から直接モデルを使用:\n"
                    f"  python main.py --model meta-llama/Llama-2-7b-hf --dataset ./Dataset/AttackDataset/prompt.json\n"
                    f"\n"
                    f"- APIモデルを使用:\n"
                    f"  export OPENAI_API_KEY='your-api-key'\n"
                    f"  python main.py --model gpt-3.5-turbo --dataset ./Dataset/AttackDataset/prompt.json\n"
                )
            else:
                error_msg += (
                    f"Hugging Face Hub からの読み込みに失敗しました。\n"
                    f"\n"
                    f"考えられる原因:\n"
                    f"1. モデル名が間違っている: '{model_path}'\n"
                    f"2. インターネット接続の問題\n"
                    f"3. Hugging Face認証が必要（プライベートリポジトリの場合）\n"
                    f"\n"
                    f"解決方法:\n"
                    f"- モデル名を確認:\n"
                    f"  有効な例: meta-llama/Llama-2-7b-hf, mistralai/Mistral-7B-Instruct-v0.2\n"
                    f"\n"
                    f"- プライベートモデルの場合、認証:\n"
                    f"  huggingface-cli login\n"
                    f"  または\n"
                    f"  export HUGGINGFACE_API_KEY='your-token'\n"
                    f"\n"
                    f"- APIモデルを使用:\n"
                    f"  export OPENAI_API_KEY='your-api-key'\n"
                    f"  python main.py --model gpt-3.5-turbo --dataset ./Dataset/AttackDataset/prompt.json\n"
                )
            
            raise RuntimeError(error_msg)
        
        # モデルの読み込み
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                low_cpu_mem_usage=True,
                local_files_only=is_local_path
            )
            
            # CPUの場合は明示的にデバイスに移動
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            print(f"[SUCCESS] モデル読み込み完了")
            print(f"[INFO] モデルパラメータ数: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
            
        except Exception as e:
            error_msg = f"モデルの読み込みに失敗: {e}\n\n"
            
            if is_local_path:
                error_msg += (
                    f"ローカルパス '{model_path}' からの読み込みに失敗しました。\n"
                    f"\n"
                    f"考えられる原因:\n"
                    f"1. モデルファイル（pytorch_model.bin または model.safetensors）が不足している\n"
                    f"2. config.json が不正または不足している\n"
                    f"3. メモリ不足（大きなモデルの場合）\n"
                    f"\n"
                    f"解決方法:\n"
                    f"- より小さいモデルを使用\n"
                    f"- Hugging Face Hub から直接使用:\n"
                    f"  python main.py --model meta-llama/Llama-2-7b-hf --dataset ./Dataset/attacks.json\n"
                    f"\n"
                    f"- APIモデルを使用（推奨）:\n"
                    f"  export OPENAI_API_KEY='your-api-key'\n"
                    f"  python main.py --model gpt-3.5-turbo --dataset ./Dataset/attacks.json\n"
                )
            else:
                error_msg += (
                    f"Hugging Face Hub からの読み込みに失敗しました。\n"
                    f"\n"
                    f"考えられる原因:\n"
                    f"1. モデル名が間違っている\n"
                    f"2. メモリ不足（大きなモデルの場合）\n"
                    f"3. インターネット接続の問題\n"
                    f"\n"
                    f"解決方法:\n"
                    f"- より小さいモデルを使用\n"
                    f"- CPUモードで実行: --device cpu\n"
                    f"- APIモデルを使用（推奨）:\n"
                    f"  export OPENAI_API_KEY='your-api-key'\n"
                    f"  python main.py --model gpt-3.5-turbo --dataset ./Dataset/attacks.json\n"
                )
            
            raise RuntimeError(error_msg)
    
    def get_model(self):
        """モデルを取得"""
        return self.model
    
    def get_tokenizer(self):
        """トークナイザーを取得"""
        return self.tokenizer
    
    def get_device(self):
        """デバイスを取得"""
        return self.device
    
    def get_model_info(self) -> dict:
        """
        モデルの詳細情報を取得
        
        Returns:
            モデル情報の辞書
        """
        # APIモデルの場合
        if self.is_api_model:
            return self.model.get_model_info()
        
        # ローカルモデルの場合（元のコード）
        info = {
            "model_path": self.model_path,
            "device": str(self.device),
            "parameters": f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M",
            "dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else "unknown",
        }
        
        # モデル設定から追加情報を取得
        if hasattr(self.model, 'config'):
            config = self.model.config
            
            # モデル名
            if hasattr(config, 'model_type'):
                info['model_type'] = config.model_type
            
            if hasattr(config, '_name_or_path'):
                info['model_name'] = config._name_or_path
            
            # アーキテクチャ情報
            if hasattr(config, 'architectures') and config.architectures:
                info['architecture'] = config.architectures[0]
            
            # レイヤー数
            if hasattr(config, 'num_hidden_layers'):
                info['num_layers'] = config.num_hidden_layers
            
            # 隠れ層のサイズ
            if hasattr(config, 'hidden_size'):
                info['hidden_size'] = config.hidden_size
            
            # Attention heads
            if hasattr(config, 'num_attention_heads'):
                info['attention_heads'] = config.num_attention_heads
            
            # 語彙サイズ
            if hasattr(config, 'vocab_size'):
                info['vocab_size'] = config.vocab_size
            
            # 最大シーケンス長
            if hasattr(config, 'max_position_embeddings'):
                info['max_length'] = config.max_position_embeddings
            elif hasattr(config, 'n_positions'):
                info['max_length'] = config.n_positions
        
        return info
    
    def print_model_info(self):
        """モデル情報を見やすく表示"""
        info = self.get_model_info()
        
        print("\n" + "="*70)
        print("評価対象モデルの詳細情報")
        print("="*70)
        
        # APIモデルの場合
        if self.is_api_model:
            print(f"モデル名: {info.get('model_name', 'N/A')}")
            print(f"モデルタイプ: {info.get('model_type', 'N/A')}")
            print(f"プロバイダー: {info.get('provider', 'N/A')}")
            print(f"API種別: {info.get('api_type', 'N/A')}")
            print("\n[INFO] これはAPIモデルです。リモートで実行されます。")
            print("="*70 + "\n")
            return
        
        # ローカルモデルの場合（元のコード）
        # 基本情報
        print(f"モデルパス: {info.get('model_path', 'N/A')}")
        if 'model_name' in info:
            print(f"モデル名: {info['model_name']}")
        if 'architecture' in info:
            print(f"アーキテクチャ: {info['architecture']}")
        if 'model_type' in info:
            print(f"モデルタイプ: {info['model_type']}")
        
        print(f"\nパラメータ数: {info.get('parameters', 'N/A')}")
        print(f"デバイス: {info.get('device', 'N/A')}")
        print(f"データ型: {info.get('dtype', 'N/A')}")
        
        # アーキテクチャ詳細
        if any(k in info for k in ['num_layers', 'hidden_size', 'attention_heads']):
            print("\nアーキテクチャ詳細:")
            if 'num_layers' in info:
                print(f"  レイヤー数: {info['num_layers']}")
            if 'hidden_size' in info:
                print(f"  隠れ層サイズ: {info['hidden_size']}")
            if 'attention_heads' in info:
                print(f"  Attentionヘッド数: {info['attention_heads']}")
            if 'vocab_size' in info:
                print(f"  語彙サイズ: {info['vocab_size']:,}")
            if 'max_length' in info:
                print(f"  最大シーケンス長: {info['max_length']:,}")
        
        print("="*70 + "\n")