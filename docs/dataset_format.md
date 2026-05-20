# データセット形式

`Dataset/` ディレクトリに配置する攻撃データのJSON形式を説明します。

## 基本形式

```json
[
  {
    "prompt": "攻撃プロンプトのテキスト",
    "attack_type": "DIRECT_OVERRIDE",
    "source": "in-the-wild"
  },
  {
    "prompt": "Base64エンコードされた攻撃プロンプト...",
    "attack_type": "ENC_EVASION",
    "source": "in-the-wild"
  }
]
```

## フィールド定義

| フィールド | 必須 | 説明 |
|-----------|------|------|
| `prompt` | ✅ | 評価対象のプロンプトテキスト |
| `attack_type` | ❌ | 既知の攻撃クラス（省略時はLLMが自動分類） |
| `source` | ❌ | データの出典 |
| `conversation_history` | ❌ | マルチターン攻撃の会話履歴（リスト形式） |

## attack_type の値

| 値 | 説明 |
|----|------|
| `BENIGN` | 正常入力 |
| `ENC_EVASION` | エンコーディング/難読化 |
| `DIRECT_OVERRIDE` | 直接的な命令上書き |
| `INTENT_CONCEAL` | 意図の隠蔽 |
| `PROGRESSIVE_MANIP` | 段階的操作（マルチターン） |
| `AUTO_ADVERSARIAL` | 自動敵対的攻撃（GCG/PAIR） |
| `UNKNOWN` | 未知の攻撃 |

## マルチターン攻撃の形式

```json
[
  {
    "prompt": "最終ターンのプロンプト",
    "attack_type": "PROGRESSIVE_MANIP",
    "conversation_history": [
      {"role": "user", "content": "ターン1のユーザー入力"},
      {"role": "assistant", "content": "ターン1のモデル応答"},
      {"role": "user", "content": "ターン2のユーザー入力"},
      {"role": "assistant", "content": "ターン2のモデル応答"}
    ]
  }
]
```

## データの入手

- **Classes 2–4**: [TrustAIRLab/in-the-wild-jailbreak-prompts](https://github.com/TrustAIRLab/in-the-wild-jailbreak-prompts)
- **Class 5**: [AIM-Intelligence/Automated-Multi-Turn-Jailbreaks](https://github.com/AIM-Intelligence/Automated-Multi-Turn-Jailbreaks)
- **Class 6**: `src/generate_auto_adversarial.py` で生成可能
