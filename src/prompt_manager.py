"""
prompt_manager.py
プロンプトの読み込み、処理、出力保存を管理するクラス
ローカルモデルとAPIモデルの両方をサポート
"""
import json
import os
import torch
from typing import Dict, List, Optional, Union
from pathlib import Path


class PromptManager:
    """
    プロンプトの読み込み、処理、出力保存を管理するクラス
    ローカルモデルとAPIモデルの両方をサポート
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[torch.device] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ):
        """
        初期化
        
        Args:
            model: 言語モデル（ローカルモデルまたはAPIModelWrapper）
            tokenizer: トークナイザー（APIモデルの場合はNone）
            device: 使用デバイス（APIモデルの場合はNone）
            max_length: 最大生成長
            temperature: サンプリング温度
            top_p: nucleus sampling parameter
            top_k: top-k sampling parameter
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
        # モデルがAPIモデルかどうかを判定
        self.is_api_model = hasattr(model, 'api_type')
        
        # 出力保存用ディレクトリ
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
    
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """
        データセットの読み込み
        
        Args:
            dataset_path: JSONファイルのパス
            
        Returns:
            攻撃データのリスト
        """
        print(f"[INFO] データセットを読み込み中: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"ファイルが見つかりません: {dataset_path}")
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # データ形式の検証と正規化
            if isinstance(data, list):
                attacks = data
            elif isinstance(data, dict):
                if "attacks" in data:
                    attacks = data["attacks"]
                elif "data" in data:
                    attacks = data["data"]
                else:
                    # 辞書の場合、単一攻撃として扱う
                    attacks = [data]
            else:
                raise ValueError("サポートされていないJSON形式です")
            
            # 各攻撃データの検証
            validated_attacks = []
            for i, attack in enumerate(attacks):
                if not isinstance(attack, dict):
                    print(f"[WARNING] 攻撃データ {i+1} は辞書ではありません。スキップします。")
                    continue
                
                if "prompt" not in attack:
                    print(f"[WARNING] 攻撃データ {i+1} に 'prompt' フィールドがありません。スキップします。")
                    continue
                
                validated_attacks.append(attack)
            
            print(f"[SUCCESS] {len(validated_attacks)}件の有効な攻撃データを読み込みました")
            return validated_attacks
            
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析エラー: {e}")
        except Exception as e:
            raise RuntimeError(f"ファイル読み込みエラー: {e}")
    
    def process_prompt(self, attack_data: Dict) -> Dict:
        """
        プロンプトをモデルに入力して出力を取得
        
        Args:
            attack_data: 攻撃データ（promptを含む）
            
        Returns:
            処理結果（prompt, output_text, logits, など）
        """
        prompt = attack_data.get("prompt", "")
        
        if not prompt:
            raise ValueError("プロンプトが空です")
        
        # モデルへの入力
        if self.is_api_model:
            output_text, logits, hidden_states = self._generate_response_api(prompt)
        else:
            output_text, logits, hidden_states = self._generate_response_local(prompt)
        
        # 結果データの構築
        result = {
            "prompt": prompt,
            "output_text": output_text,
            "logits": logits,
            "hidden_states": hidden_states,
            "metadata": {
                "attack_type": attack_data.get("attack_type", "Unknown"),
                "conversation_history": attack_data.get("conversation_history", None),
                "max_length": self.max_length,
                "temperature": self.temperature,
                "is_api_model": self.is_api_model
            }
        }
        
        return result
    
    def _generate_response_local(self, prompt: str) -> tuple:
        """
        ローカルモデルから応答を生成
        
        Args:
            prompt: 入力プロンプト
            
        Returns:
            (生成テキスト, ロジット, 隠れ層の状態)のタプル
        """
        try:
            # トークン化
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            ).to(self.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 生成されたトークンのデコード
            generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # ロジットの取得（最初のトークンのスコア）
            logits = None
            if outputs.scores and len(outputs.scores) > 0:
                logits = outputs.scores[0][0]  # [vocab_size]
            
            # 隠れ層の状態（最後の層）
            hidden_states = None
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                # hidden_states は tuple of tuples
                # 最後の生成ステップの最後の層を取得
                if len(outputs.hidden_states) > 0:
                    last_step = outputs.hidden_states[-1]  # 最後の生成ステップ
                    if len(last_step) > 0:
                        hidden_states = last_step[-1]  # 最後の層
            
            return generated_text, logits, hidden_states
            
        except Exception as e:
            print(f"[ERROR] ローカルモデル応答生成エラー: {e}")
            return "", None, None
    
    def _generate_response_api(self, prompt: str) -> tuple:
        """
        APIモデルから応答を生成
        
        Args:
            prompt: 入力プロンプト
            
        Returns:
            (生成テキスト, ロジット, 隠れ層の状態)のタプル
            注: APIモデルの場合、ロジットと隠れ層は取得できないのでNone
        """
        try:
            # APIモデルで生成
            result = self.model.generate(
                prompt=prompt,
                max_tokens=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            # エラーチェック
            if "error" in result:
                print(f"[ERROR] API呼び出しエラー: {result['error']}")
                return "", None, None
            
            generated_text = result.get("text", "")
            
            # APIモデルからはロジットと隠れ層は取得できない
            return generated_text, None, None
            
        except Exception as e:
            print(f"[ERROR] APIモデル応答生成エラー: {e}")
            return "", None, None
    
    def save_output(self, output_data: Dict, filename: str = None) -> str:
        """
        出力データをJSON形式で保存
        
        Args:
            output_data: 保存するデータ
            filename: 保存ファイル名（Noneの場合は自動生成）
            
        Returns:
            保存先のファイルパス
        """
        if filename is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"output_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # ロジットとhidden_statesはテンソルなので保存形式を変換
        save_data = {
            "prompt": output_data["prompt"],
            "output_text": output_data["output_text"],
            "metadata": output_data["metadata"]
        }
        
        # ロジットの保存（上位k個のみ）- ローカルモデルのみ
        if output_data.get("logits") is not None:
            logits = output_data["logits"]
            top_k = 10
            top_values, top_indices = torch.topk(logits, top_k)
            save_data["logits"] = {
                "top_values": top_values.cpu().tolist(),
                "top_indices": top_indices.cpu().tolist()
            }
        
        # 隠れ層の状態（統計情報のみ）- ローカルモデルのみ
        if output_data.get("hidden_states") is not None:
            hidden_states = output_data["hidden_states"]
            save_data["hidden_states_stats"] = {
                "shape": list(hidden_states.shape),
                "mean": hidden_states.mean().item(),
                "std": hidden_states.std().item(),
                "min": hidden_states.min().item(),
                "max": hidden_states.max().item()
            }
        
        # JSON保存
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        return str(filepath)
    
    def save_batch_outputs(self, outputs: List[Dict], filename: str = "batch_outputs.json") -> str:
        """
        複数の出力を一括保存
        
        Args:
            outputs: 出力データのリスト
            filename: 保存ファイル名
            
        Returns:
            保存先のファイルパス
        """
        filepath = self.output_dir / filename
        
        batch_data = []
        for output in outputs:
            save_item = {
                "prompt": output["prompt"],
                "output_text": output["output_text"],
                "metadata": output["metadata"]
            }
            
            if output.get("logits") is not None:
                logits = output["logits"]
                top_k = 10
                top_values, top_indices = torch.topk(logits, top_k)
                save_item["logits"] = {
                    "top_values": top_values.cpu().tolist(),
                    "top_indices": top_indices.cpu().tolist()
                }
            
            batch_data.append(save_item)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=2)
        
        return str(filepath)
    
    def get_output_directory(self) -> Path:
        """出力ディレクトリのパスを取得"""
        return self.output_dir
    
    def set_generation_params(
        self,
        max_length: int = None,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None
    ):
        """
        生成パラメータの更新
        
        Args:
            max_length: 最大生成長
            temperature: サンプリング温度
            top_p: nucleus sampling parameter
            top_k: top-k sampling parameter
        """
        if max_length is not None:
            self.max_length = max_length
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if top_k is not None:
            self.top_k = top_k
        
        print(f"[INFO] 生成パラメータを更新:")
        print(f"  - max_length: {self.max_length}")
        print(f"  - temperature: {self.temperature}")
        print(f"  - top_p: {self.top_p}")
        print(f"  - top_k: {self.top_k}")