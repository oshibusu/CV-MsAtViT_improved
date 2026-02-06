import matplotlib.pyplot as plt
import re
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Plot learning curve from extracted log file.")
    parser.add_argument("input_file", help="Path to input extracted epochs text file")
    parser.add_argument("output_plot", help="Path to output plot image file")
    args = parser.parse_args()

    input_file = args.input_file
    output_plot = args.output_plot

    losses = []
    accuracies = []
    epochs = []

    print(f"Reading {input_file}...")

    try:
        with open(input_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        sys.exit(1)

    epoch_counter = 0

    for line in lines:
        # Skip ETA lines as they are duplicates or intermediate states
        if "ETA:" in line:
            continue
            
        # Extract loss and accuracy
        # Pattern example: ... loss: 0.3896 - accuracy: 0.8630
        loss_match = re.search(r"loss:\s*([\d\.]+)", line)
        acc_match = re.search(r"accuracy:\s*([\d\.]+)", line)
        
        if loss_match and acc_match:
            epoch_counter += 1
            losses.append(float(loss_match.group(1)))
            accuracies.append(float(acc_match.group(1)))
            epochs.append(epoch_counter)

    print(f"Found {len(epochs)} valid epoch entries.")

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Loss', color='tab:blue')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Accuracy', color='tab:orange')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"Saved plot to {output_plot}")

if __name__ == "__main__":
    main()
