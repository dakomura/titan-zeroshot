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
```
directory/
└── 20x_512px_0px_overlap/
    └── slide_features_titan/
        ├── sample1.h5
        ├── sample2.h5
        └── ...
```

### テーブルファイル
TSVまたはCSVファイルで、`sample`列にサンプル名、その他の列にカテゴリ変数を含む。 