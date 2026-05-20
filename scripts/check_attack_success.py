#!/usr/bin/env python3
"""
攻撃成功判定のテストスクリプト
"""
import json
import sys

def check_attack_success_detection(results_file: str):
    """
    結果ファイルから攻撃成功判定の状態を確認
    
    Args:
        results_file: 結果JSONファイルのパス
    """
    print(f"結果ファイルを確認中: {results_file}")
    print("="*70)
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        total = len(results)
        with_detection = 0
        without_detection = 0
        benign_count = 0
        
        print(f"\n総結果数: {total}")
        print("\n攻撃成功判定の状態:")
        print("-"*70)
        
        for i, result in enumerate(results, 1):
            attack_type = result.get('detected_attack_type', 'UNKNOWN')
            success_detection = result.get('attack_success_detection')
            
            if attack_type == 'BENIGN':
                benign_count += 1
                status = "スキップ（BENIGN）"
            elif success_detection is not None:
                with_detection += 1
                attack_success = success_detection.get('attack_success', False)
                confidence = success_detection.get('confidence', 0.0)
                votes = success_detection.get('votes', {})
                status = f"{'成功' if attack_success else '失敗'} (信頼度: {confidence:.2f}, 投票: {votes.get('success', 0)}/{votes.get('total', 0)})"
            else:
                without_detection += 1
                status = "❌ NULL"
            
            print(f"#{i:2d} [{attack_type:20s}]: {status}")
        
        print("\n" + "="*70)
        print("サマリー:")
        print(f"  BENIGN（スキップ）: {benign_count}")
        print(f"  攻撃成功判定あり: {with_detection}")
        print(f"  攻撃成功判定なし（NULL）: {without_detection}")
        
        if without_detection > benign_count:
            print("\n⚠️  警告: BENIGN以外で攻撃成功判定がNULLの結果があります")
            print("   修正後のevaluation.pyを使用していることを確認してください")
            return False
        else:
            print("\n✅ すべての攻撃（BENIGN以外）で攻撃成功判定が実行されています")
            return True
            
    except FileNotFoundError:
        print(f"❌ エラー: ファイルが見つかりません: {results_file}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ エラー: JSON解析エラー: {e}")
        return False
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python check_attack_success.py <results.json>")
        print("\n例:")
        print("  python check_attack_success.py ./results/results_20251219_235056.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    success = check_attack_success_detection(results_file)
    sys.exit(0 if success else 1)