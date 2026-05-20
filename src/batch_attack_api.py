#!/usr/bin/env python3
"""
バッチ攻撃実行システム - GCG, PAIR, AMJ対応（HuggingFace & API対応版）

JSONファイルから複数の攻撃指示を読み込み、自動的に順次実行します。

対応モデル:
    - HuggingFace: meta-llama/Llama-2-7b-hf, mistralai/Mistral-7B-v0.1, etc.
    - OpenAI API: gpt-4, gpt-3.5-turbo, gpt-4-turbo (PAIR/AMJのみ)
    - Anthropic API: claude-3-5-sonnet-20241022, claude-3-opus-20240229 (PAIR/AMJのみ)

使用方法:
    # HuggingFace モデル
    python batch_attack.py --config ./Dataset/Harmbench_100data.json --method gcg
    
    # OpenAI API
    export OPENAI_API_KEY="sk-proj-..."
    python batch_attack.py --config ./Dataset/Harmbench_100data.json --method pair
    
    # Anthropic API
    export ANTHROPIC_API_KEY="sk-ant-..."
    python batch_attack.py --config ./Dataset/Harmbench_100data.json --method amj

    
    python batch_attack_api.py --config ./Dataset/Harmbench_10data.json --method pair
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import traceback

# カレントディレクトリとsrcディレクトリをPythonパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# GCG攻撃のインポート（複数の可能性を試す）
GCG_AVAILABLE = False
try:
    from gcg_attack import GCGAttackWithEvaluation
    GCG_AVAILABLE = True
    print("[SUCCESS] GCG攻撃が利用可能です（gcg_attack.py）")
except ImportError:
    try:
        from src.gcg_attack import GCGAttackWithEvaluation
        GCG_AVAILABLE = True
        print("[SUCCESS] GCG攻撃が利用可能です（src/gcg_attack.py）")
    except ImportError:
        print("[WARNING] GCG攻撃が利用できません")

# PAIR攻撃のインポート（複数の可能性を試す）
PAIR_AVAILABLE = False
try:
    from pair_attack import PAIRAttack
    PAIR_AVAILABLE = True
    print("[SUCCESS] PAIR攻撃が利用可能です（pair_attack.py）")
except ImportError:
    try:
        from src.pair_attack import PAIRAttack
        PAIR_AVAILABLE = True
        print("[SUCCESS] PAIR攻撃が利用可能です（src/pair_attack.py）")
    except ImportError as e:
        print(f"[WARNING] PAIR攻撃が利用できません: {e}")

# AMJ攻撃のインポート（複数の可能性を試す）
AMJ_AVAILABLE = False
try:
    from amj_attack import AMJAttack
    AMJ_AVAILABLE = True
    print("[SUCCESS] AMJ攻撃が利用可能です（amj_attack.py）")
except ImportError:
    try:
        from src.amj_attack import AMJAttack
        AMJ_AVAILABLE = True
        print("[SUCCESS] AMJ攻撃が利用可能です（src/amj_attack.py）")
    except ImportError as e:
        print(f"[WARNING] AMJ攻撃が利用できません: {e}")


def detect_model_type(model_name: str) -> str:
    """
    モデル名から種類を自動判定
    
    Args:
        model_name: モデル名
        
    Returns:
        "huggingface", "openai", or "anthropic"
    """
    model_lower = model_name.lower()
    
    # OpenAI API
    if model_lower.startswith("gpt-") or model_lower.startswith("o1-"):
        return "openai"
    
    # Anthropic API
    if model_lower.startswith("claude-"):
        return "anthropic"
    
    # HuggingFace モデル（デフォルト）
    return "huggingface"


class BatchAttackExecutor:
    """バッチ攻撃実行システム - HuggingFace & API対応"""
    
    def __init__(
        self,
        config_path: str,
        method: str,
        output_dir: str,
        attack_ids: Optional[List[str]] = None
    ):
        """
        Args:
            config_path: 攻撃設定JSONファイルのパス
            method: 攻撃手法（gcg, pair, amj）
            output_dir: 結果出力ディレクトリ
            attack_ids: 実行する攻撃のID（Noneの場合は全て実行）
        """
        self.config_path = config_path
        self.method = method.lower()
        self.output_dir = output_dir
        self.attack_ids = attack_ids
        
        # 出力ディレクトリの作成
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 設定の読み込み
        print(f"[INFO] Loading configuration from: {config_path}")
        self.config = self._load_config()
        
        # 実行する攻撃のフィルタリング
        self.attacks = self._filter_attacks()
        
        print(f"[INFO] Method: {self.method}")
        print(f"[INFO] Total attacks to execute: {len(self.attacks)}")
        print(f"[INFO] Output directory: {output_dir}")
    
    def _load_config(self) -> Dict:
        """設定ファイルを読み込む"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 必須フィールドのチェック
            if "attacks" not in config:
                raise ValueError("設定ファイルに'attacks'フィールドがありません")
            
            return config
        
        except FileNotFoundError:
            raise FileNotFoundError(f"設定ファイルが見つかりません: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSONの解析に失敗しました: {e}")
    
    def _filter_attacks(self) -> List[Dict]:
        """実行する攻撃をフィルタリング"""
        attacks = []
        
        for attack in self.config["attacks"]:
            # IDのチェック
            if self.attack_ids and attack.get("id") not in self.attack_ids:
                continue
            
            # enabledフラグのチェック
            if not attack.get("enabled", True):
                print(f"[SKIP] Attack '{attack.get('id')}' is disabled")
                continue
            
            # 必須フィールドのチェック
            if "goal" not in attack:
                print(f"[WARNING] Attack '{attack.get('id')}' has no 'goal' field, skipping")
                continue
            
            attacks.append(attack)
        
        return attacks
    
    def _get_global_config(self, key: str, default=None):
        """グローバル設定を取得"""
        return self.config.get("global_config", {}).get(key, default)
    
    def _get_attack_config(self, attack: Dict, key: str, default=None):
        """攻撃固有の設定を取得（グローバル設定よりも優先）"""
        return attack.get(key, self._get_global_config(key, default))
    
    def execute_gcg_attack(self, attack: Dict) -> Dict:
        """GCG攻撃を実行（HuggingFaceモデルのみ対応）"""
        if not GCG_AVAILABLE:
            raise RuntimeError("GCG攻撃が利用できません")
        
        print(f"\n{'='*70}")
        print(f"Executing GCG Attack: {attack.get('id', 'unknown')}")
        print(f"{'='*70}")
        
        # パラメータの取得
        target_model = self._get_attack_config(attack, "target_model", "meta-llama/Llama-2-7b-hf")
        device = self._get_attack_config(attack, "device", "cuda")
        num_iterations = self._get_attack_config(attack, "num_iterations", 100)
        suffix_length = self._get_attack_config(attack, "suffix_length", 20)
        use_evaluator = self._get_attack_config(attack, "use_evaluator", True)
        
        # モデルタイプの判定
        model_type = detect_model_type(target_model)
        
        print(f"[CONFIG] Target Model: {target_model} (Type: {model_type})")
        print(f"[CONFIG] Device: {device}")
        print(f"[CONFIG] Iterations: {num_iterations}")
        print(f"[CONFIG] Suffix Length: {suffix_length}")
        print(f"[CONFIG] Use Evaluator: {use_evaluator}")
        
        # GCG攻撃はHuggingFaceモデルのみ対応
        if model_type != "huggingface":
            print(f"[WARNING] GCG attack only supports HuggingFace models")
            print(f"[WARNING] API models (OpenAI, Anthropic) are not supported for GCG")
        
        # GCG攻撃の初期化
        gcg = GCGAttackWithEvaluation(
            model_name=target_model,
            device=device,
            suffix_length=suffix_length,
            use_evaluator=use_evaluator,
        )
        
        # 攻撃の実行
        query = attack["goal"]
        target = attack.get("target", "Sure, here's how to do that:")
        
        suffix, history = gcg.optimize(
            query=query,
            target=target,
            num_iterations=num_iterations,
            verbose=True
        )
        
        return {
            "attack_id": attack.get("id", "unknown"),
            "method": "gcg",
            "goal": query,
            "target_model": target_model,
            "model_type": model_type,
            "optimized_suffix": suffix,
            "num_iterations": num_iterations,
            "history": history,
            "status": "completed"
        }
    
    def execute_pair_attack(self, attack: Dict) -> Dict:
        """PAIR攻撃を実行（HuggingFace & API対応）"""
        if not PAIR_AVAILABLE:
            raise RuntimeError("PAIR攻撃が利用できません")
        
        print(f"\n{'='*70}")
        print(f"Executing PAIR Attack: {attack.get('id', 'unknown')}")
        print(f"{'='*70}")
        
        # パラメータの取得
        target_model = self._get_attack_config(attack, "target_model", "meta-llama/Llama-2-7b-hf")
        attacker_model = self._get_attack_config(attack, "attacker_model", "gpt-4")
        device = self._get_attack_config(attack, "device", "cuda")
        num_iterations = self._get_attack_config(attack, "num_iterations", 10)
        use_evaluator = self._get_attack_config(attack, "use_evaluator", True)
        
        # モデルタイプの判定
        target_model_type = detect_model_type(target_model)
        attacker_model_type = detect_model_type(attacker_model)
        
        print(f"[CONFIG] Target Model: {target_model} (Type: {target_model_type})")
        print(f"[CONFIG] Attacker Model: {attacker_model} (Type: {attacker_model_type})")
        print(f"[CONFIG] Device: {device}")
        print(f"[CONFIG] Iterations: {num_iterations}")
        print(f"[CONFIG] Use Evaluator: {use_evaluator}")
        
        # PAIR攻撃の初期化（シンプルなインターフェース）
        pair = PAIRAttack(
            target_model_name=target_model,
            attacker_model=attacker_model,
            device=device,
            use_evaluator=use_evaluator,
            use_env_keys=True
        )
        
        # 攻撃の実行
        goal = attack["goal"]
        
        best_prompt, history = pair.attack(
            goal=goal,
            num_iterations=num_iterations,
            verbose=True
        )
        
        return {
            "attack_id": attack.get("id", "unknown"),
            "method": "pair",
            "goal": goal,
            "target_model": target_model,
            "target_model_type": target_model_type,
            "attacker_model": attacker_model,
            "attacker_model_type": attacker_model_type,
            "best_prompt": best_prompt,
            "num_iterations": num_iterations,
            "history": history,
            "status": "completed"
        }
    
    def execute_amj_attack(self, attack: Dict) -> Dict:
        """AMJ攻撃を実行（HuggingFace & API対応）"""
        if not AMJ_AVAILABLE:
            raise RuntimeError("AMJ攻撃が利用できません")
        
        print(f"\n{'='*70}")
        print(f"Executing AMJ Attack: {attack.get('id', 'unknown')}")
        print(f"{'='*70}")
        
        # パラメータの取得
        target_model = self._get_attack_config(attack, "target_model", "meta-llama/Llama-2-7b-hf")
        attacker_model = self._get_attack_config(attack, "attacker_model", "gpt-4")
        device = self._get_attack_config(attack, "device", "cuda")
        num_turns = self._get_attack_config(attack, "num_turns", 5)
        use_evaluator = self._get_attack_config(attack, "use_evaluator", True)
        
        # モデルタイプの判定
        target_model_type = detect_model_type(target_model)
        attacker_model_type = detect_model_type(attacker_model)
        
        print(f"[CONFIG] Target Model: {target_model} (Type: {target_model_type})")
        print(f"[CONFIG] Attacker Model: {attacker_model} (Type: {attacker_model_type})")
        print(f"[CONFIG] Device: {device}")
        print(f"[CONFIG] Turns: {num_turns}")
        print(f"[CONFIG] Use Evaluator: {use_evaluator}")
        
        # AMJ攻撃の初期化（シンプルなインターフェース）
        amj = AMJAttack(
            target_model_name=target_model,
            attacker_model=attacker_model,
            device=device,
            use_evaluator=use_evaluator,
            use_env_keys=True
        )
        
        # 攻撃の実行
        goal = attack["goal"]
        
        conversation_history, turn_details = amj.attack(
            goal=goal,
            num_turns=num_turns,
            verbose=True
        )
        
        return {
            "attack_id": attack.get("id", "unknown"),
            "method": "amj",
            "goal": goal,
            "target_model": target_model,
            "target_model_type": target_model_type,
            "attacker_model": attacker_model,
            "attacker_model_type": attacker_model_type,
            "conversation_history": conversation_history,
            "turn_details": turn_details,
            "num_turns": num_turns,
            "status": "completed"
        }
    
    def execute_attack(self, attack: Dict) -> Dict:
        """攻撃を実行（手法に応じて振り分け）"""
        try:
            if self.method == "gcg":
                return self.execute_gcg_attack(attack)
            elif self.method == "pair":
                return self.execute_pair_attack(attack)
            elif self.method == "amj":
                return self.execute_amj_attack(attack)
            else:
                raise ValueError(f"Unknown attack method: {self.method}")
        
        except Exception as e:
            print(f"[ERROR] Attack failed: {e}")
            traceback.print_exc()
            
            return {
                "attack_id": attack.get("id", "unknown"),
                "method": self.method,
                "goal": attack.get("goal", "unknown"),
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def save_result(self, result: Dict):
        """結果を保存"""
        attack_id = result.get("attack_id", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ファイル名の生成
        filename = f"{self.method}_{attack_id}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # 結果の保存
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Result saved to: {filepath}")
    
    def execute_all(self) -> List[Dict]:
        """すべての攻撃を実行"""
        print(f"\n{'='*70}")
        print(f"Starting Batch Attack Execution")
        print(f"{'='*70}")
        print(f"Method: {self.method.upper()}")
        print(f"Total attacks: {len(self.attacks)}")
        print(f"{'='*70}\n")
        
        results = []
        
        for i, attack in enumerate(self.attacks, 1):
            attack_id = attack.get("id", f"attack_{i}")
            
            print(f"\n{'='*70}")
            print(f"Attack {i}/{len(self.attacks)}: {attack_id}")
            print(f"Goal: {attack.get('goal', 'unknown')}")
            print(f"{'='*70}\n")
            
            # 攻撃の実行
            result = self.execute_attack(attack)
            
            # 結果の保存
            self.save_result(result)
            
            # 結果のリストに追加
            results.append(result)
            
            print(f"\n[INFO] Attack {i}/{len(self.attacks)} completed")
            print(f"[INFO] Status: {result.get('status', 'unknown')}")
        
        # サマリーの作成
        self._create_summary(results)
        
        return results
    
    def _create_summary(self, results: List[Dict]):
        """実行サマリーを作成・保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.output_dir, f"summary_{self.method}_{timestamp}.json")
        
        # 統計の計算
        total = len(results)
        completed = sum(1 for r in results if r.get("status") == "completed")
        failed = sum(1 for r in results if r.get("status") == "failed")
        
        # モデルタイプ別の統計
        model_types = {}
        for result in results:
            if result.get("status") == "completed":
                model_type = result.get("target_model_type", result.get("model_type", "unknown"))
                model_types[model_type] = model_types.get(model_type, 0) + 1
        
        # 成功判定（最後の結果が成功しているか）
        successful_attacks = 0
        for result in results:
            if result.get("status") == "completed":
                history = result.get("history", [])
                turn_details = result.get("turn_details", [])
                
                if history and isinstance(history, list) and len(history) > 0:
                    last_item = history[-1]
                    if isinstance(last_item, dict):
                        attack_success = last_item.get("attack_success", {})
                        if isinstance(attack_success, dict) and attack_success.get("successful", False):
                            successful_attacks += 1
                
                elif turn_details and isinstance(turn_details, list) and len(turn_details) > 0:
                    last_turn = turn_details[-1]
                    if isinstance(last_turn, dict):
                        attack_success = last_turn.get("attack_success", {})
                        if isinstance(attack_success, dict) and attack_success.get("successful", False):
                            successful_attacks += 1
        
        summary = {
            "timestamp": timestamp,
            "method": self.method,
            "config_file": self.config_path,
            "output_directory": self.output_dir,
            "statistics": {
                "total_attacks": total,
                "completed": completed,
                "failed": failed,
                "successful_attacks": successful_attacks,
                "success_rate": f"{successful_attacks / total * 100:.1f}%" if total > 0 else "N/A",
                "model_types": model_types
            },
            "results": results
        }
        
        # サマリーの保存
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # コンソール出力
        print(f"\n{'='*70}")
        print(f"Batch Execution Summary")
        print(f"{'='*70}")
        print(f"Total Attacks: {total}")
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        print(f"Successful Attacks: {successful_attacks}")
        print(f"Success Rate: {summary['statistics']['success_rate']}")
        if model_types:
            print(f"Model Types: {model_types}")
        print(f"{'='*70}")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="バッチ攻撃実行システム - GCG, PAIR, AMJ対応（HuggingFace & API対応版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # HuggingFace モデルで攻撃
  python batch_attack.py --config ./Dataset/Harmbench_100data.json --method gcg
  
  # OpenAI API で攻撃（PAIR/AMJ）
  export OPENAI_API_KEY="sk-proj-..."
  python batch_attack.py --config ./Dataset/Harmbench_100data.json --method pair
  
  # Anthropic API で攻撃（PAIR/AMJ）
  export ANTHROPIC_API_KEY="sk-ant-..."
  python batch_attack.py --config ./Dataset/Harmbench_100data.json --method amj

対応モデル:
  HuggingFace: meta-llama/Llama-2-7b-hf, mistralai/Mistral-7B-v0.1, etc.
  OpenAI: gpt-4, gpt-3.5-turbo, gpt-4-turbo (PAIR/AMJのみ)
  Anthropic: claude-3-5-sonnet-20241022, claude-3-opus-20240229 (PAIR/AMJのみ)
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="攻撃設定JSONファイルのパス"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["gcg", "pair", "amj"],
        help="攻撃手法（gcg, pair, amj）"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="batch_results",
        help="結果出力ディレクトリ（デフォルト: batch_results）"
    )
    
    parser.add_argument(
        "--attack-ids",
        type=str,
        default=None,
        help="実行する攻撃のIDをカンマ区切りで指定（例: attack_1,attack_3）"
    )
    
    args = parser.parse_args()
    
    # 攻撃IDのパース
    attack_ids = None
    if args.attack_ids:
        attack_ids = [aid.strip() for aid in args.attack_ids.split(",")]
    
    # バッチ実行システムの初期化
    executor = BatchAttackExecutor(
        config_path=args.config,
        method=args.method,
        output_dir=args.output_dir,
        attack_ids=attack_ids
    )
    
    # すべての攻撃を実行
    results = executor.execute_all()
    
    print(f"\n[INFO] All attacks completed!")
    print(f"[INFO] Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()