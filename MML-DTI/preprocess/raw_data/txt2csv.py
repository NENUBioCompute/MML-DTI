import csv
import os


def txt_to_csv(txt_file_path, csv_output_path):
    """
    Convert Davis.txt file to CSV containing only SMILES, Protein, and Y columns

    Parameters:
    txt_file_path: Full path to input txt file (e.g., "C:/data/Davis.txt")
    csv_output_path: Full path to output csv file (e.g., "C:/data/Davis_filtered.csv")
    """
    # Check if input file exists
    if not os.path.exists(txt_file_path):
        raise FileNotFoundError(f"Input file not found: {txt_file_path}")

    # Open txt file for reading and csv file for writing
    with open(txt_file_path, 'r', encoding='utf-8') as txt_file, \
            open(csv_output_path, 'w', newline='', encoding='utf-8') as csv_file:

        # Initialize CSV writer with column headers
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["SMILES", "Protein", "Y"])  # Write header row

        # Process each line in txt file
        for line_num, line in enumerate(txt_file, start=1):
            # Remove leading/trailing whitespace
            clean_line = line.strip()
            # Skip empty lines
            if not clean_line:
                continue

            # Split columns by whitespace
            columns = clean_line.split()

            # Validate row has exactly 5 columns (original format: drugid, protid, SMILES, Protein, Y)
            if len(columns) != 5:
                print(f"Warning: Line {line_num} has abnormal format (columns={len(columns)}), skipping: {clean_line}")
                continue

            # Extract target columns (index 2=SMILES, index 3=Protein, index 4=Y)
            smiles = columns[2]
            protein = columns[3]
            y_value = columns[4]

            # Write to CSV
            csv_writer.writerow([smiles, protein, y_value])

    print(f"Conversion completed! CSV file saved to: {csv_output_path}")


# ------------------- Modify the following parameters according to your file paths -------------------
# Input txt file path (absolute or relative path)
INPUT_TXT_PATH = "KIBA.txt"
# Output csv file path (customize save location and filename)
OUTPUT_CSV_PATH = "../../data/KIBA/fulldata.csv"
# ----------------------------------------------------------------------------------------------------

# Execute conversion
if __name__ == "__main__":
    txt_to_csv(INPUT_TXT_PATH, OUTPUT_CSV_PATH)