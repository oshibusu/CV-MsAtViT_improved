import re
import os

# Class definitions from analyze_results.py
CLASS_NAMES_BALTRUM = [
    "Background", "Tidal flat", "Water", "Coastal shrub", "Dense, high vegetation",
    "White dune", "Peat bog", "Grey dunes", "Couch grass", "Upper salt marsh",
    "Lower salt marsh", "Sand", "Settlement"
]

CLASS_NAMES_FL_T = [
    "Background", "Water", "Forest", "Lucerne", "Grass", "Rapeseed", "Beet",
    "Potatoes", "Peas", "Stem Beans", "Bare Soil", "Wheat", "Wheat 2",
    "Wheat 3", "Barley", "Buildings"
]

CLASS_NAMES_SF = [
    "Background", "Bare Soil", "Mountain", "Water", "Urban", "Vegetation"
]

def get_expected_classes(filename):
    if "SF" in filename:
        return CLASS_NAMES_SF
    elif "FL_T" in filename:
        return CLASS_NAMES_FL_T
    elif "Baltrum" in filename:
        return CLASS_NAMES_BALTRUM
    return []

def parse_report_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Split into sections based on "Analyzing:"
    sections = content.split("Analyzing: ")
    results = []
    
    for section in sections[1:]: # Skip preamble
        lines = section.split('\n')
        filename = lines[0].strip()
        
        # Find Classification Report
        try:
            start_idx = lines.index("--- Classification Report ---")
        except ValueError:
            continue
            
        # Parse table
        # Format: ClassName precision recall f1-score support
        # Skip header lines
        table_lines = []
        capture = False
        for line in lines[start_idx:]:
            if "precision" in line and "recall" in line:
                capture = True
                continue
            if not capture:
                continue
            if line.strip() == "":
                continue
            if "accuracy" in line:
                break
                
            table_lines.append(line)
            
        # Parse logic
        parsed_data = []
        for line in table_lines:
            parts = line.split()
            if len(parts) < 5: continue
            
            # Extract numbers (last 4)
            support = parts[-1]
            f1 = parts[-2]
            recall = parts[-3]
            precision = parts[-4]
            
            # Everything before is class name
            class_name = " ".join(parts[:-4])
            parsed_data.append({
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support
            })
            
        results.append({
            "filename": filename,
            "data": parsed_data
        })
        
    return results

def generate_latex(results, dataset_name, expected_classes):
    print(f"\\section{{Results for {dataset_name}}}")
    
    # Verify order
    # Expected classes usually start from index 1 (Background is 0 and often skipped)
    # But let's check what's in the data
    
    for res in results:
        print(f"% Table for {res['filename']}")
        print("\\begin{table}[htbp]")
        print("    \\centering")
        safe_filename = res['filename'].replace('_', '\\_')
        print(f"    \\caption{{Classification Report for {safe_filename}}}")
        print("    \\begin{tabular}{lcccc}")
        print("        \\toprule")
        print("        Class & Precision & Recall & F1-score & Support \\\\")
        print("        \\midrule")
        
        data_classes = [d['class'] for d in res['data']]
        
        # Check order
        # Filter expected classes to those present in data (matches by name)
        # Note: Report might match partial names or use different spacing, but usually mapped from CLASS_NAMES
        
        is_order_correct = True
        
        # Map back to indices in expected_classes
        present_indices = []
        for d in data_classes:
            try:
                idx = expected_classes.index(d)
                present_indices.append(idx)
            except ValueError:
                # Fallback check for stripping
                found = False
                for i, exp in enumerate(expected_classes):
                    if exp.strip() == d.strip():
                        present_indices.append(i)
                        found = True
                        break
                if not found:
                    print(f"% WARNING: Class '{d}' not found in expected list!")
                    is_order_correct = False
        
        # Check if indices are sorted
        if present_indices != sorted(present_indices):
            is_order_correct = False
            print(f"% WARNING: Class order in report does NOT match expected index order!")
            print(f"% Report order indices: {present_indices}")
        else:
            print(f"% Class order verified: Correct (Indices: {present_indices})")

        for row in res['data']:
            print(f"        {row['class']} & {row['precision']} & {row['recall']} & {row['f1']} & {row['support']} \\\\")
            
        print("        \\bottomrule")
        print("    \\end{tabular}")
        print("    \\label{tab:" + res['filename'].replace('.', '_').replace(' ', '_') + "}")
        print("\\end{table}")
        print("\n")

def main():
    files = [
        ("results/SF_analysis_report.txt", "San Francisco"),
        ("results/FL_T_analysis_report.txt", "Flevoland"),
        ("results/Baltrum_analysis_report.txt", "Baltrum")
    ]
    
    for filepath, name in files:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        results = parse_report_file(filepath)
        expected = get_expected_classes(filepath)
        generate_latex(results, name, expected)

if __name__ == "__main__":
    main()
