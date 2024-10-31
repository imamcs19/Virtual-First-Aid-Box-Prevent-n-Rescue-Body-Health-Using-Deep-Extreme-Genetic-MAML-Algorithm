import pandas as pd
import os

# Define constants for the number of rows to take from each sheet
n_data_model_1 = 3
n_data_model_2 = 2

# Define file paths
output_file = "dataset/one_hot_encoded_data.xlsx"
file_path = "dataset_init/dataset_v5.xlsx"

# Ensure the output directory exists
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# Load the input and target data sheets from the output file
input_data = pd.read_excel(output_file, sheet_name="OneHotEncodedDataInput")
target_data = pd.read_excel(output_file, sheet_name="OneHotEncodedDataTarget")

# Create datasets for each required sheet in the new file
# sheet1_data = pd.concat([input_data.iloc[:n_data_model_1, :], target_data.iloc[:n_data_model_1, :]], axis=0, ignore_index=True)
# sheet2_data = pd.concat([input_data.iloc[-n_data_model_2:, :], target_data.iloc[-n_data_model_2:, :]], axis=0, ignore_index=True)
# combined_data = pd.concat([input_data, target_data], axis=0, ignore_index=True)

# Create datasets for each required sheet by concatenating input and target side-by-side
sheet1_data = pd.concat([input_data.iloc[:n_data_model_1, :], target_data.iloc[:n_data_model_1, :]], axis=1)
sheet2_data = pd.concat([input_data.iloc[-n_data_model_2:, :], target_data.iloc[-n_data_model_2:, :]], axis=1)
combined_data = pd.concat([input_data, target_data], axis=1)

# Save these datasets into a new Excel file with the specified sheet names
with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
    sheet1_data.to_excel(writer, sheet_name="Sheet1-KM-SAR-Tiny-Reg", index=False)
    sheet2_data.to_excel(writer, sheet_name="Sheet2-M-SAK-T-Tiny-Reg", index=False)
    combined_data.to_excel(writer, sheet_name="Comb-KMT-Tiny-Reg", index=False)

print(f"File '{file_path}' created with sheets: 'Sheet1-KM-SAR-Tiny-Reg', 'Sheet2-M-SAK-T-Tiny-Reg', and 'Comb-KMT-Tiny-Reg'.")