import re
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Extract training log lines (loss/accuracy) from log file.")
    parser.add_argument("input_file", help="Path to input log file")
    parser.add_argument("output_file", help="Path to output extracted text file")
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    print(f"Reading {input_file}...")

    try:
        with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        sys.exit(1)

    extracted = []
    # Pattern: Look for lines like "192/192 [====...] - 1s 5ms/step - loss: 0.123 - accuracy: 0.987"
    # We look for "loss:" and "accuracy:" and a progress bar structure
    # Generic pattern: digits/digits [ ... ] ... loss: ... accuracy: ...
    pattern = r"\d+/\d+\s*\[.*\]\s+-\s+.*loss:\s+[\d\.]+\s+-\s+accuracy:\s+[\d\.]+"
    
    # Fallback pattern for some keras versions: just look for loss and accuracy if step info is missing or formatted differently
    # But usually we want the final step of the epoch which has the bar.
    
    # Also handle the case where the progress bar uses unicode or different chars
    
    for line in lines:
        if "loss:" in line and "accuracy:" in line:
            # Check if it looks like a progress bar line (starts with N/M)
            match = re.match(r"\s*(\d+)/(\d+)", line)
            if match:
                current_step = int(match.group(1))
                total_steps = int(match.group(2))
                
                # Only keep the final step of the epoch
                if current_step == total_steps:
                    # Exclude intermediate "ETA" lines, keep the final summary line
                    if "ETA:" not in line:
                         extracted.append(line)

    print(f"Found {len(extracted)} matching lines.")

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(extracted)

    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
