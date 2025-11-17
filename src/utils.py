import matplotlib.pyplot as plt
import os

def plot_results(data_dict, title, xlabel, ylabel, file_path, secondary_y_keys=None):
    """
    주어진 데이터 딕셔너리를 사용하여 그래프를 그리고 파일로 저장합니다.
    각 라인에 다른 색상을 할당하고, 보조 Y축을 지원합니다.
    """
    if secondary_y_keys is None:
        secondary_y_keys = []

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 색상 리스트
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Primary Y-axis
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    
    primary_keys = [k for k in data_dict.keys() if k not in secondary_y_keys]
    for i, key in enumerate(primary_keys):
        ax1.plot(data_dict[key], label=key, color=colors[i % len(colors)])
    ax1.tick_params(axis='y')

    # Secondary Y-axis
    lines, labels = ax1.get_legend_handles_labels()
    
    if secondary_y_keys:
        ax2 = ax1.twinx()
        secondary_keys = secondary_y_keys
        for i, key in enumerate(secondary_keys):
            color_idx = len(primary_keys) + i
            ax2.plot(data_dict[key], label=f'{key} (right)', color=colors[color_idx % len(colors)], linestyle='--')
        ax2.set_ylabel(f'{", ".join(secondary_keys)} Value')
        ax2.tick_params(axis='y')
        
        sec_lines, sec_labels = ax2.get_legend_handles_labels()
        lines += sec_lines
        labels += sec_labels

    fig.suptitle(title)
    ax1.legend(lines, labels, loc='best')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()
    print(f"Plot saved to {file_path}")


