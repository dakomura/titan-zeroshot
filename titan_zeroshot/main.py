#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import numpy as np
import h5py
import umap
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_distances
import seaborn as sns
import logging
from datetime import datetime

def load_titan_features(feature_dir, logger):
    """titan特徴量を読み込む"""
    features = {}
    
    for filename in os.listdir(feature_dir):
        if filename.endswith('.h5'):
            sample_name = filename.replace('.h5', '')
            filepath = os.path.join(feature_dir, filename)
            
            with h5py.File(filepath, 'r') as f:
                # h5ファイルの構造を確認して適切にデータを読み込む
                # 一般的にはメインのデータセットキーがあるはず
                keys = list(f.keys())
                if len(keys) == 1:
                    feature_vector = f[keys[0]][:]
                else:
                    # 複数キーがある場合は最初のキーを使用
                    logger.warning(f"Multiple keys found in {filename}: {keys}")
                    feature_vector = f[keys[0]][:]
                
                features[sample_name] = feature_vector
    
    return features

def create_umap_plot(features_dict, table_df, color_attr, output_path, logger):
    """UMAPで次元削減してプロット"""
    # 特徴量とサンプル名を整理
    sample_names = []
    feature_matrix = []
    
    for sample_name, feature_vector in features_dict.items():
        if sample_name in table_df['sample'].values:
            sample_names.append(sample_name)
            feature_matrix.append(feature_vector)
    
    feature_matrix = np.array(feature_matrix)
    logger.info(f"Feature matrix shape: {feature_matrix.shape}")
    
    # UMAP実行
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(feature_matrix)
    
    # プロット用のDataFrame作成
    plot_df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'sample': sample_names
    })
    
    # テーブルデータとマージ
    plot_df = plot_df.merge(table_df[['sample', color_attr]], on='sample', how='left')
    
    # プロット
    plt.figure(figsize=(10, 8))
    unique_categories = plot_df[color_attr].unique()
    colors = sns.color_palette("husl", len(unique_categories))
    
    for i, category in enumerate(unique_categories):
        mask = plot_df[color_attr] == category
        plt.scatter(plot_df.loc[mask, 'UMAP1'], 
                   plot_df.loc[mask, 'UMAP2'], 
                   c=[colors[i]], 
                   label=str(category), 
                   alpha=0.7)
    
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title(f'UMAP visualization colored by {color_attr}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"UMAP plot saved to: {output_path}")

def evaluate_knn_classification(features_dict, table_df, target_attr, k, output_path, logger):
    """k-NNによるLOOCV評価"""
    # データ準備
    sample_names = []
    feature_matrix = []
    labels = []
    
    for sample_name, feature_vector in features_dict.items():
        if sample_name in table_df['sample'].values:
            sample_names.append(sample_name)
            feature_matrix.append(feature_vector)
            label = table_df[table_df['sample'] == sample_name][target_attr].iloc[0]
            labels.append(label)
    
    feature_matrix = np.array(feature_matrix)
    labels = np.array(labels)
    sample_names = np.array(sample_names)
    
    logger.info(f"Classification data shape: {feature_matrix.shape}")
    logger.info(f"Number of samples: {len(labels)}")
    
    # コサイン距離を使用するk-NN分類器
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    
    # LOOCV実行
    loo = LeaveOneOut()
    predictions = []
    true_labels = []
    neighbor_info = []
    
    for train_idx, test_idx in loo.split(feature_matrix):
        X_train, X_test = feature_matrix[train_idx], feature_matrix[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        train_names = sample_names[train_idx]
        test_name = sample_names[test_idx][0]
        
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        
        # 近傍情報を取得
        distances, indices = knn.kneighbors(X_test)
        neighbor_names = train_names[indices[0]]
        neighbor_labels = y_train[indices[0]]
        neighbor_distances = distances[0]
        
        # 近傍情報を保存
        neighbors_data = {
            'test_sample': test_name,
            'true_label': y_test[0],
            'predicted_label': pred[0],
            'neighbors': []
        }
        
        for i in range(k):
            neighbors_data['neighbors'].append({
                'sample': neighbor_names[i],
                'label': neighbor_labels[i],
                'distance': neighbor_distances[i]
            })
        
        neighbor_info.append(neighbors_data)
        predictions.extend(pred)
        true_labels.extend(y_test)
    
    # 結果分析
    unique_categories = np.unique(labels)
    results = []
    
    for category in unique_categories:
        mask = labels == category
        category_count = np.sum(mask)
        
        # その属性のサンプルでの正解率
        category_mask = np.array(true_labels) == category
        if np.sum(category_mask) > 0:
            category_accuracy = accuracy_score(
                np.array(true_labels)[category_mask], 
                np.array(predictions)[category_mask]
            )
        else:
            category_accuracy = 0.0
        
        results.append({
            'category': category,
            'sample_count': category_count,
            'accuracy': category_accuracy
        })
    
    # 全体の精度
    overall_accuracy = accuracy_score(true_labels, predictions)
    
    # 結果をファイルに保存
    results_df = pd.DataFrame(results)
    
    with open(output_path, 'w') as f:
        f.write(f"k-NN Classification Results (k={k})\n")
        f.write(f"Target attribute: {target_attr}\n")
        f.write(f"Overall accuracy: {overall_accuracy:.4f}\n\n")
        f.write("Category-wise results:\n")
        f.write(results_df.to_string(index=False))
        f.write(f"\n\nTotal samples: {len(labels)}")
    
    # 近傍情報を別ファイルに保存
    neighbors_output = output_path.replace('.txt', '_neighbors.txt')
    with open(neighbors_output, 'w') as f:
        f.write(f"k-NN Neighbors Information (k={k})\n")
        f.write(f"Target attribute: {target_attr}\n\n")
        
        for info in neighbor_info:
            f.write(f"Test Sample: {info['test_sample']}\n")
            f.write(f"True Label: {info['true_label']}\n")
            f.write(f"Predicted Label: {info['predicted_label']}\n")
            f.write(f"Correct: {'Yes' if info['true_label'] == info['predicted_label'] else 'No'}\n")
            f.write("Nearest Neighbors:\n")
            
            for i, neighbor in enumerate(info['neighbors'], 1):
                f.write(f"  {i}. {neighbor['sample']} (label: {neighbor['label']}, distance: {neighbor['distance']:.4f})\n")
            
            f.write("\n" + "-"*50 + "\n\n")
    
    logger.info(f"Classification results saved to: {output_path}")
    logger.info(f"Neighbor information saved to: {neighbors_output}")
    logger.info(f"Overall accuracy: {overall_accuracy:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Titan特徴量の分析（UMAP可視化とk-NN分類）')
    parser.add_argument('directory', help='ベースディレクトリ（A）のパス')
    parser.add_argument('table_file', help='テーブルファイル（TSV/CSV）のパス')
    parser.add_argument('color_attr', help='プロット時の色分け属性名')
    parser.add_argument('--classify_attr', help='分類対象の属性名（指定しない場合は色分け属性と同じ）')
    parser.add_argument('--k', type=int, default=5, help='k-NNのk値（デフォルト: 5）')
    parser.add_argument('--output_prefix', default='output', help='出力ファイルのプレフィックス（デフォルト: output）')
    
    args = parser.parse_args()
    
    # 分類属性が指定されていない場合は色分け属性を使用
    classify_attr = args.classify_attr if args.classify_attr else args.color_attr
    
    # ログファイルの設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{args.output_prefix}_analysis_{timestamp}.log"
    
    # ロガーの設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=== Titan特徴量分析開始 ===")
    logger.info(f"ベースディレクトリ: {args.directory}")
    logger.info(f"テーブルファイル: {args.table_file}")
    logger.info(f"色分け属性: {args.color_attr}")
    logger.info(f"分類属性: {classify_attr}")
    logger.info(f"k-NNのk値: {args.k}")
    logger.info(f"出力プレフィックス: {args.output_prefix}")
    logger.info(f"ログファイル: {log_filename}")
    
    # パス設定
    feature_dir = os.path.join(args.directory, '20x_512px_0px_overlap', 'slide_features_titan')
    umap_output = f"{args.output_prefix}_umap_{args.color_attr}.png"
    classification_output = f"{args.output_prefix}_classification_{classify_attr}_k{args.k}.txt"
    
    # 入力チェック
    if not os.path.exists(feature_dir):
        logger.error(f"Feature directory not found: {feature_dir}")
        return
    
    if not os.path.exists(args.table_file):
        logger.error(f"Table file not found: {args.table_file}")
        return
    
    # データ読み込み
    logger.info("Loading titan features...")
    features_dict = load_titan_features(feature_dir, logger)
    logger.info(f"Loaded {len(features_dict)} feature files")
    
    logger.info("Loading table data...")
    if args.table_file.endswith('.tsv'):
        table_df = pd.read_csv(args.table_file, sep='\t')
    else:
        table_df = pd.read_csv(args.table_file)
    
    logger.info(f"Table shape: {table_df.shape}")
    logger.info(f"Available columns: {list(table_df.columns)}")
    
    # 属性の存在チェック
    if args.color_attr not in table_df.columns:
        logger.error(f"Color attribute '{args.color_attr}' not found in table")
        return
    
    if classify_attr not in table_df.columns:
        logger.error(f"Classification attribute '{classify_attr}' not found in table")
        return
    
    # サンプル統計情報の出力
    logger.info("=== サンプル統計情報 ===")
    logger.info(f"テーブルファイルのサンプル数: {len(table_df)}")
    logger.info(f"利用可能なh5ファイル数: {len(features_dict)}")
    
    # 欠損サンプルのチェックと記録
    table_samples = set(table_df['sample'].values)
    available_samples = set(features_dict.keys())
    missing_samples = table_samples - available_samples
    common_samples = table_samples & available_samples
    
    logger.info(f"テーブルとh5ファイルの共通サンプル数: {len(common_samples)}")
    logger.info(f"欠損サンプル数: {len(missing_samples)}")
    
    if missing_samples:
        logger.warning("=== 欠損サンプル一覧 ===")
        for sample in sorted(missing_samples):
            logger.warning(f"Missing h5 file for sample: {sample}")
    
    # 各属性の統計情報
    logger.info("=== 属性別統計情報 ===")
    for attr in [args.color_attr, classify_attr]:
        if attr in table_df.columns:
            value_counts = table_df[attr].value_counts()
            logger.info(f"{attr}属性の分布:")
            for value, count in value_counts.items():
                logger.info(f"  {value}: {count}サンプル")
    
    # 解析対象サンプルの統計情報
    analysis_samples = [s for s in table_df['sample'].values if s in features_dict]
    analysis_df = table_df[table_df['sample'].isin(analysis_samples)]
    
    logger.info("=== 解析対象サンプル統計 ===")
    logger.info(f"解析対象サンプル数: {len(analysis_samples)}")
    
    for attr in [args.color_attr, classify_attr]:
        if attr in analysis_df.columns:
            value_counts = analysis_df[attr].value_counts()
            logger.info(f"解析対象の{attr}属性の分布:")
            for value, count in value_counts.items():
                logger.info(f"  {value}: {count}サンプル")
    
    # 処理1: UMAP可視化
    logger.info("Creating UMAP visualization...")
    create_umap_plot(features_dict, table_df, args.color_attr, umap_output, logger)
    
    # 処理2: k-NN分類評価
    logger.info("Evaluating k-NN classification...")
    evaluate_knn_classification(features_dict, table_df, classify_attr, args.k, classification_output, logger)
    
    logger.info("=== 分析完了 ===")
    logger.info(f"ログファイル: {log_filename}")
    logger.info(f"UMAPプロット: {umap_output}")
    logger.info(f"分類結果: {classification_output}")
    logger.info(f"近傍情報: {classification_output.replace('.txt', '_neighbors.txt')}")

if __name__ == "__main__":
    main() 