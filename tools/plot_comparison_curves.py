import matplotlib.pyplot as plt
import re
import argparse
import os

def parse_epochs(filepath):
    """Parses loss and accuracy from an epoch log file."""
    losses = []
    accuracies = []
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found.")
        return [], []
    
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        if "ETA:" in line: continue
        loss_match = re.search(r"loss:\s*([\d\.]+)", line)
        acc_match = re.search(r"accuracy:\s*([\d\.]+)", line)
        if loss_match and acc_match:
            losses.append(float(loss_match.group(1)))
            accuracies.append(float(acc_match.group(1)))
    return losses, accuracies

def main():
    parser = argparse.ArgumentParser(description="Plot comparison of two training logs.")
    parser.add_argument("--previous", required=True, help="Path to previous (default) epoch log")
    parser.add_argument("--proposed", required=True, help="Path to proposed epoch log")
    parser.add_argument("--output", required=True, help="Output plot filename")
    parser.add_argument("--title", default="Training Comparison", help="Plot title")
    args = parser.parse_args()

    p_loss, p_acc = parse_epochs(args.previous)
    ours_loss, ours_acc = parse_epochs(args.proposed)

    plt.figure(figsize=(16, 12)) # Wide and Tall

    # Plot Loss
    plt.subplot(1, 2, 1) # Left
    if p_loss:
        plt.plot(range(1, len(p_loss) + 1), p_loss, label='Previous', color='tab:blue', linestyle='--', linewidth=2.5)
    if ours_loss:
        plt.plot(range(1, len(ours_loss) + 1), ours_loss, label='Proposed', color='tab:orange', linewidth=2.5)
    plt.title('Training Loss', fontsize=24)
    plt.xlabel('Epoch', fontsize=22)
    plt.ylabel('Loss', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=20)

    # Plot Accuracy
    plt.subplot(1, 2, 2) # Right
    if p_acc:
        plt.plot(range(1, len(p_acc) + 1), p_acc, label='Previous', color='tab:blue', linestyle='--', linewidth=2.5)
    if ours_acc:
        plt.plot(range(1, len(ours_acc) + 1), ours_acc, label='Proposed', color='tab:orange', linewidth=2.5)
    plt.title('Training Accuracy', fontsize=24)
    plt.xlabel('Epoch', fontsize=22)
    plt.ylabel('Accuracy', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=20)
    plt.ylim(0, 1.05)

    plt.suptitle(args.title, fontsize=28)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    plt.savefig(args.output, dpi=200)
    plt.close()
    print(f"Comparison plot saved to {args.output}")

if __name__ == "__main__":
    main()
