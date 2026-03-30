import pandas as pd
import os
import glob

input_folder = 'data/raw-dataset/'
output_folder = 'data/cleaned-dataset/'
os.makedirs(output_folder, exist_ok=True)

csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

print(f"Found {len(csv_files)} files to process...")

for input_path in csv_files:
    file_base = os.path.splitext(os.path.basename(input_path))[0]
    output_name = os.path.join(output_folder, f"{file_base}_cleaned.csv")
    
    df = pd.read_csv(input_path)
    initial_row_count = len(df)
    
    # FILTER SUCCESSFUL ENTRIES
    df_cleaned = df[df['success'] == 1].copy()
    final_row_count = len(df_cleaned)
    rows_removed = initial_row_count - final_row_count

    # LABEL BINARIZATION
    # Boredom, Confusion, Frustration: 1 or higher becomes 1, else 0
    cols_to_bin_1 = ['Boredom', 'Confusion', 'Frustration']
    for col in cols_to_bin_1:
        if col in df_cleaned.columns:
            df_cleaned[col] = (df_cleaned[col] >= 1).astype(int)
    
    # Engagement: 3 becomes 1, else 0
    if 'Engagement' in df_cleaned.columns:
        df_cleaned['Engagement'] = (df_cleaned['Engagement'] >= 3).astype(int)

    # ROW REMOVAL
    # Only 3D facial landmarks are left (millimeters)
    remove_prefixes = ('gaze', 'eye_lmk', 'AU', 'MP', 'left_hand', 'right_hand', 'x', 'y', 'pose')
    
    to_remove = []
    for col in df_cleaned.columns:
        if col.startswith(remove_prefixes):
            to_remove.append(col)

    df_final = df_cleaned.drop(columns=to_remove)

    df_final.to_csv(output_name, index=False)
    print(f"Processed: {file_base} | Columns removed: {len(to_remove)} | Rows removed: {rows_removed} | Columns remaining: {df_final.shape[1]}")

print("\nAll files have been cleaned and saved to:", output_folder)