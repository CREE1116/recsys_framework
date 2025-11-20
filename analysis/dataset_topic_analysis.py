import os
import re
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import sys

# 프로젝트의 src 및 analysis 폴더를 python 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.utils import load_item_metadata, get_analysis_output_path, AnalysisReport

def clean_text(text):
    """텍스트에서 특수문자, 숫자, 연도를 제거하는 등 전처리를 수행합니다."""
    text = re.sub(r'\(\d{4}\)', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

def plot_top_words(model, feature_names, n_top_words, title, save_path):
    """LDA 모델의 토픽별 상위 단어를 시각화하고 토픽 데이터를 반환합니다."""
    n_topics = model.n_components
    fig, axes = plt.subplots(n_topics, 1, figsize=(10, 5 * n_topics), sharex=True)
    fig.suptitle(title, fontsize=20, fontweight='bold')
    if n_topics == 1: axes = [axes]
    
    topic_data = []
    for topic_idx, topic in enumerate(model.components_):
        top_feat_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_feat = [feature_names[i] for i in top_feat_idx]
        top_feat_val = topic[top_feat_idx]
        topic_data.append(", ".join(top_feat))
        
        ax = axes[topic_idx]
        ax.barh(np.arange(n_top_words), top_feat_val, align='center', color='skyblue', ecolor='black')
        ax.set_yticks(np.arange(n_top_words))
        ax.set_yticklabels(top_feat, fontsize=12)
        ax.invert_yaxis()
        ax.set_title(f'Topic #{topic_idx}', fontsize=16)
        ax.set_xlabel('Weight')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    
    return pd.DataFrame({'Topic ID': range(n_topics), 'Top Keywords': topic_data})

def run_dataset_topic_analysis(dataset_config_path, n_topics=5, n_top_words=10):
    """
    데이터셋의 아이템 메타데이터에 대해 LDA 토픽 모델링을 수행하고 리포트를 생성합니다.
    """
    print(f"\nRunning Dataset Topic Analysis for: {dataset_config_path}")
    
    with open(dataset_config_path, 'r') as f:
        config = yaml.safe_load(f)
    dataset_name = config['dataset_name']
    
    output_path = get_analysis_output_path(dataset_name)
    report = AnalysisReport(f"Dataset Topic Analysis: {dataset_name}", output_path)

    metadata_df = load_item_metadata(dataset_name, config['data_path'])
    if metadata_df.empty:
        report.add_text("Metadata DataFrame is empty. Skipping analysis.")
        report.save("dataset_topic_report.md")
        return

    metadata_df['genre_str'] = metadata_df['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    metadata_df['text_for_lda'] = metadata_df['title'].apply(clean_text) + ' ' + metadata_df['genre_str']
    documents = metadata_df['text_for_lda'].tolist()

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=3, ngram_range=(1, 2))
    try:
        dtm = vectorizer.fit_transform(documents)
    except ValueError as e:
        report.add_text(f"Could not vectorize documents. Maybe corpus is too small? Error: {e}")
        report.save("dataset_topic_report.md")
        return
        
    feature_names = vectorizer.get_feature_names_out()

    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method='online', n_jobs=-1)
    lda_model.fit(dtm)

    figure_filename = "topic_keywords.png"
    save_path = os.path.join(output_path, figure_filename)
    
    topics_df = plot_top_words(
        lda_model, 
        feature_names, 
        n_top_words,
        f'LDA Topics for {dataset_name.upper()}',
        save_path
    )
    
    report.add_section("1. Discovered Topics Summary", level=2)
    report.add_table(topics_df)
    
    report.add_section("2. Top Keywords per Topic", level=2)
    report.add_figure(figure_filename, "Bar chart showing top keywords for each discovered topic.")
    
    report.save("dataset_topic_report.md")


if __name__ == '__main__':
    EXPERIMENTS_TO_RUN = [
        {'dataset_config': 'configs/dataset/ml1m.yaml', 'n_topics': 8, 'n_top_words': 10},
        {'dataset_config': 'configs/dataset/ml100k.yaml', 'n_topics': 5, 'n_top_words': 10},
    ]

    for exp_config in EXPERIMENTS_TO_RUN:
        if os.path.exists(exp_config['dataset_config']):
            run_dataset_topic_analysis(**exp_config)
        else:
            print(f"[Warning] Dataset config not found, skipping: {exp_config['dataset_config']}")
