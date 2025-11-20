import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def plot_losses(experiment_dir, output_dir=None):
    """
    실험 디렉토리에서 손실 기록을 로드하고 시각화합니다.
    """
    print(f"Plotting losses for experiment: {experiment_dir}")

    losses_history_path = os.path.join(experiment_dir, 'losses_history.json')

    if not os.path.exists(losses_history_path):
        print(f"Error: 'losses_history.json' not found in {experiment_dir}")
        return

    with open(losses_history_path, 'r') as f:
        losses_data = json.load(f)

    epochs = range(1, len(losses_data['total_loss']) + 1)

    if output_dir is None:
        output_dir = experiment_dir

    # 1. 총 손실 플롯
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses_data['total_loss'], label='Total Loss', color='blue')
    plt.title(f'Total Training Loss Over Epochs\n(Experiment: {os.path.basename(experiment_dir)})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    total_loss_output_path = os.path.join(output_dir, 'total_loss_plot.png')
    plt.savefig(total_loss_output_path)
    print(f"Total loss plot saved to {total_loss_output_path}")
    plt.close()

    # 2. 개별 손실 구성 요소 플롯
    component_losses = {k: v for k, v in losses_data.items() if k.startswith('loss_') and k != 'total_loss'}
    
    if component_losses:
        plt.figure(figsize=(12, 8))
        for loss_name, loss_values in component_losses.items():
            plt.plot(epochs, loss_values, label=loss_name)
        
        plt.title(f'Individual Training Loss Components Over Epochs\n(Experiment: {os.path.basename(experiment_dir)})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.grid(True)
        plt.legend()
        component_losses_output_path = os.path.join(output_dir, 'component_losses_plot.png')
        plt.savefig(component_losses_output_path)
        print(f"Component losses plot saved to {component_losses_output_path}")
        plt.close()
    else:
        print("No individual loss components found to plot (only total_loss).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot training loss history from an experiment directory.")
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Path to the experiment directory (containing losses_history.json).')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the output plots. Defaults to the experiment directory.')
    
    args = parser.parse_args()
    
    plot_losses(args.exp_dir, output_dir=args.output_dir)
