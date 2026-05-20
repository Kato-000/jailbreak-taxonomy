#!/usr/bin/env python3
"""
攻撃成功率の可視化スクリプト
結果JSONから攻撃成功率のグラフを生成
"""
import json
import sys
import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # バックエンド設定
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlibがインストールされていません")
    print("グラフ生成には以下を実行してください:")
    print("  pip install matplotlib")


def load_results(filepath: str) -> dict:
    """結果JSONファイルを読み込み"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_text_summary(data: dict):
    """テキスト形式でサマリーを表示"""
    summary = data.get('summary', {})
    attack_summary = summary.get('attack_success_summary')
    
    if not attack_summary:
        print("攻撃成功率データが見つかりません")
        return
    
    print("\n" + "="*70)
    print("攻撃成功率サマリー".center(70))
    print("="*70)
    
    # 総合成功率
    overall_rate = attack_summary['overall_success_rate'] * 100
    print(f"\n総合攻撃成功率: {overall_rate:.2f}%")
    print(f"  テスト数: {attack_summary['total_attacks_tested']}")
    print(f"  成功: {attack_summary['successful_attacks']}")
    print(f"  失敗: {attack_summary['failed_attacks']}")
    
    # 攻撃タイプ別
    print(f"\n攻撃タイプ別成功率:")
    print("-"*70)
    
    success_by_type = attack_summary['success_rate_by_type']
    sorted_types = sorted(success_by_type.items(), 
                         key=lambda x: x[1]['success_rate'], 
                         reverse=True)
    
    for attack_type, stats in sorted_types:
        risk_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}[stats['risk_level']]
        print(f"{risk_emoji} {attack_type:20s}: {stats['success_rate']*100:6.2f}% "
              f"({stats['successes']}/{stats['attempts']})")
    
    print("="*70 + "\n")


def create_bar_chart(data: dict, output_path: str):
    """攻撃タイプ別成功率の棒グラフを生成"""
    if not HAS_MATPLOTLIB:
        print("matplotlibが必要です")
        return False
    
    attack_summary = data['summary'].get('attack_success_summary')
    if not attack_summary:
        print("攻撃成功率データが見つかりません")
        return False
    
    success_by_type = attack_summary['success_rate_by_type']
    
    # データの準備
    types = []
    rates = []
    colors = []
    
    # 成功率でソート
    sorted_types = sorted(success_by_type.items(), 
                         key=lambda x: x[1]['success_rate'], 
                         reverse=True)
    
    for attack_type, stats in sorted_types:
        types.append(attack_type)
        rates.append(stats['success_rate'] * 100)
        
        # リスクレベルに応じて色を設定
        if stats['risk_level'] == 'high':
            colors.append('#ff6b6b')  # 赤
        elif stats['risk_level'] == 'medium':
            colors.append('#ffa94d')  # オレンジ
        else:
            colors.append('#51cf66')  # 緑
    
    # グラフ作成
    plt.figure(figsize=(12, 6))
    bars = plt.bar(types, rates, color=colors, alpha=0.8, edgecolor='black')
    
    # 総合成功率の横線
    overall_rate = attack_summary['overall_success_rate'] * 100
    plt.axhline(y=overall_rate, color='blue', linestyle='--', 
                linewidth=2, label=f'総合成功率 ({overall_rate:.1f}%)')
    
    # ラベルと装飾
    plt.xlabel('攻撃タイプ', fontsize=12, fontweight='bold')
    plt.ylabel('成功率 (%)', fontsize=12, fontweight='bold')
    plt.title('攻撃タイプ別成功率', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # バーの上に値を表示
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 棒グラフを保存: {output_path}")
    plt.close()
    return True


def create_pie_chart(data: dict, output_path: str):
    """総合的な成功/失敗の円グラフを生成"""
    if not HAS_MATPLOTLIB:
        print("matplotlibが必要です")
        return False
    
    attack_summary = data['summary'].get('attack_success_summary')
    if not attack_summary:
        print("攻撃成功率データが見つかりません")
        return False
    
    # データの準備
    labels = ['攻撃成功', '防御成功']
    sizes = [
        attack_summary['successful_attacks'],
        attack_summary['failed_attacks']
    ]
    colors = ['#ff6b6b', '#51cf66']
    explode = (0.05, 0)  # 成功部分を少し強調
    
    # グラフ作成
    plt.figure(figsize=(10, 8))
    wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, 
                                        colors=colors, autopct='%1.1f%%',
                                        shadow=True, startangle=90,
                                        textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    # パーセンテージを白く太字に
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')
    
    overall_rate = attack_summary['overall_success_rate'] * 100
    plt.title(f'総合攻撃成功率: {overall_rate:.1f}%\n'
              f'(成功: {attack_summary["successful_attacks"]}件 / '
              f'テスト: {attack_summary["total_attacks_tested"]}件)',
              fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 円グラフを保存: {output_path}")
    plt.close()
    return True


def create_comparison_chart(filepaths: list, output_path: str):
    """複数の結果を比較するグラフを生成"""
    if not HAS_MATPLOTLIB:
        print("matplotlibが必要です")
        return False
    
    # 各ファイルのデータを読み込み
    all_data = []
    labels = []
    
    for i, filepath in enumerate(filepaths):
        data = load_results(filepath)
        attack_summary = data['summary'].get('attack_success_summary')
        if attack_summary:
            all_data.append(attack_summary)
            # ファイル名をラベルに使用
            labels.append(Path(filepath).stem)
    
    if len(all_data) < 2:
        print("比較するには2つ以上の結果ファイルが必要です")
        return False
    
    # グラフ作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # サブプロット1: 総合成功率の比較
    overall_rates = [d['overall_success_rate'] * 100 for d in all_data]
    colors_overall = ['#ff6b6b' if r >= 70 else '#ffa94d' if r >= 40 else '#51cf66' 
                     for r in overall_rates]
    
    bars1 = ax1.bar(labels, overall_rates, color=colors_overall, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('成功率 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('総合攻撃成功率の比較', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # バーの上に値を表示
    for bar, rate in zip(bars1, overall_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # サブプロット2: 攻撃タイプ別の比較（最初のファイルの攻撃タイプ順）
    attack_types = list(all_data[0]['success_rate_by_type'].keys())
    x = range(len(attack_types))
    width = 0.8 / len(all_data)
    
    for i, (data, label) in enumerate(zip(all_data, labels)):
        rates = []
        for attack_type in attack_types:
            if attack_type in data['success_rate_by_type']:
                rates.append(data['success_rate_by_type'][attack_type]['success_rate'] * 100)
            else:
                rates.append(0)
        
        offset = (i - len(all_data)/2 + 0.5) * width
        ax2.bar([p + offset for p in x], rates, width, label=label, alpha=0.8)
    
    ax2.set_ylabel('成功率 (%)', fontsize=12, fontweight='bold')
    ax2.set_title('攻撃タイプ別成功率の比較', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(attack_types, rotation=45, ha='right')
    ax2.set_ylim(0, 105)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 比較グラフを保存: {output_path}")
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description='攻撃成功率の可視化')
    parser.add_argument('results', nargs='+', help='結果JSONファイル（1つ以上）')
    parser.add_argument('--output-dir', '-o', default='./visualizations',
                       help='出力ディレクトリ（デフォルト: ./visualizations）')
    parser.add_argument('--bar-chart', action='store_true',
                       help='棒グラフを生成（最初のファイルのみ）')
    parser.add_argument('--pie-chart', action='store_true',
                       help='円グラフを生成（最初のファイルのみ）')
    parser.add_argument('--comparison', action='store_true',
                       help='比較グラフを生成（複数ファイル必要）')
    parser.add_argument('--all', action='store_true',
                       help='すべてのグラフを生成')
    
    args = parser.parse_args()
    
    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 最初のファイルのテキストサマリーを常に表示
    data = load_results(args.results[0])
    print_text_summary(data)
    
    # グラフ生成
    if not HAS_MATPLOTLIB:
        print("\nグラフ生成をスキップします（matplotlib未インストール）")
        return
    
    if args.all or args.bar_chart:
        bar_output = output_dir / f"{Path(args.results[0]).stem}_bar.png"
        create_bar_chart(data, str(bar_output))
    
    if args.all or args.pie_chart:
        pie_output = output_dir / f"{Path(args.results[0]).stem}_pie.png"
        create_pie_chart(data, str(pie_output))
    
    if (args.all or args.comparison) and len(args.results) >= 2:
        comparison_output = output_dir / "comparison.png"
        create_comparison_chart(args.results, str(comparison_output))
    elif args.comparison and len(args.results) < 2:
        print("比較グラフには2つ以上の結果ファイルが必要です")
    
    if not (args.bar_chart or args.pie_chart or args.comparison or args.all):
        print("\nヒント: グラフを生成するには --all, --bar-chart, --pie-chart, --comparison のいずれかを指定してください")


if __name__ == "__main__":
    main()