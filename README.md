# titan-zeroshot

Titan特徴量の分析ツール（UMAP可視化とk-NN分類）

TITAN特徴量は [TRIDENT](https://github.com/mahmoodlab/TRIDENT) ライブラリで計算されたものを使用します。

## インストール

```bash
uv pip install git+https://github.com/dakomura/titan-zeroshot.git
```

または通常のpipでも可能：
```bash
pip install git+https://github.com/dakomura/titan-zeroshot.git
```

## 使用方法

```bash
titan-zeroshot /path/to/directory table.csv color_attribute --classify_attr classification_attribute --k 5 --output_prefix result
```

### 引数

- `directory`: ベースディレクトリA
- `table_file`: TSV/CSVファイル
- `color_attr`: プロット時の色分け属性
- `--classify_attr`: 分類対象属性（省略時は色分け属性と同じ）
- `--k`: k-NNのk値（デフォルト5）
- `--output_prefix`: 出力ファイル名のプレフィックス

### 出力ファイル

- `{prefix}_umap_{color_attr}.png`: UMAPプロット
- `{prefix}_classification_{classify_attr}_k{k}.txt`: 分類結果
- `{prefix}_classification_{classify_attr}_k{k}_neighbors.txt`: 各サンプルの近傍詳細情報

## データ形式

### 入力ディレクトリ構造

ベースディレクトリAを指定した場合、以下のような構成になっている必要があります：

```
A/
└── 20x_512px_0px_overlap/
    └── slide_features_titan/
        ├── sample1.h5
        ├── sample2.h5
        ├── sample3.h5
        └── ...
```

各`.h5`ファイルは、[TRIDENT](https://github.com/mahmoodlab/TRIDENT)ライブラリで計算されたTitan特徴量を含んでいます。ファイル名は`{サンプル名}.h5`の形式で、テーブルファイルの`sample`列と一致する必要があります。

### テーブルファイル

TSVまたはCSVファイルで、以下の形式である必要があります：

- `sample`列：サンプル名（.h5ファイル名と一致）
- その他の列：カテゴリ変数（色分けや分類に使用）

例：
```csv
sample,diagnosis,stage
sample1,normal,early
sample2,cancer,late
sample3,normal,early
``` 