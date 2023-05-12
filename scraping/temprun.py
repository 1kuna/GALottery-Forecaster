import csv
import os
import glob

# Define function to get full file path
def get_file_path(*subdirs, filename=None):
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, *subdirs)
    if filename is not None:
        full_path = os.path.join(full_path, filename)
    full_path = full_path.replace("/", "\\")
    return full_path

csv_folder = get_file_path("csv")

# Combine all the CSV files into one CSV file
csv_files = [f for f in glob.glob(f"{csv_folder}\\*.csv")]

temp_combined_file = f"{csv_folder}/temp_combined.csv"
header = []

with open(temp_combined_file, "w", newline="") as combined_csv:
    writer = csv.writer(combined_csv)
    for csv_file in csv_files:
        with open(csv_file, "r") as individual_csv:
            reader = csv.reader(individual_csv)
            header_skipped = False
            for row in reader:
                if not header_skipped:
                    header_skipped = True
                    if not header:
                        header = row
                    continue
                writer.writerow(row)

# Add the header to the first line of the combined file
combined_file = f"{csv_folder}/combined.csv"
with open(temp_combined_file, "r") as temp_csv, open(combined_file, "w", newline="") as combined_csv:
    writer = csv.writer(combined_csv)
    writer.writerow(header)
    reader = csv.reader(temp_csv)
    for row in reader:
        writer.writerow(row)

# Remove the temporary combined file
os.remove(temp_combined_file)

# # Remove the CSV files
# for csv_file in csv_files:
#     os.remove(f"{csv_folder}\\{csv_file}")