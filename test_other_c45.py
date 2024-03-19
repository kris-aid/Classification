#!/usr/bin/env python
import pdb
from c45 import C45
import csv
import os

def split_csv(input_file):
    # Create a folder to store the split files
    output_folder = "data"
    os.makedirs(output_folder, exist_ok=True)

    with open(input_file, 'r') as file:
        lines = file.readlines()

        if len(lines) < 1:
            print("CSV file is empty.")
            return

        # Extract the first line (header)
        header = lines[0].strip().split(',')
        new_names = [f"item{i}" for i in range(1, len(header) + 1)]
        modified_header = [f"{name} : continuous\n" for name in new_names]
        # Save file with header
        with open(os.path.join(output_folder, 'subset.names'), 'w', newline='') as with_header_file:
            with_header_file.writelines("0.0, 1.0\n")
            with_header_file.writelines(modified_header[:-1])


        # Save file without header
        with open(os.path.join(output_folder, 'subset.data'), 'w', newline='') as without_header_file:
            without_header_file.writelines(lines[1:])  # Exclude last line

        print("Split files created successfully.")

# Specify the input CSV file
input_csv_file = "C:/Users/Usuario/Desktop/dec_tree\subsets_variable_length/subset_1_34_df.csv"

# Call the function to split the CSV
split_csv(input_csv_file)

c1 = C45("data/subset.data", "data/subset.names")
c1.fetchData()
c1.preprocessData()
c1.generateTree()
c1.printTree()
