import pandas as pd
def remove_col(csv_file, column_name, output_dir):
    df = pd.read_csv(csv_file)
    if column_name not in df.columns:
        return
    df = df.drop(columns=[column_name])
    df.to_csv(output_dir)

def remove_index(csv_file, output_dir):
    df = pd.read_csv(csv_file)
    df = df.drop(df.columns[0], axis=1)
    df.to_csv(output_dir, index =False)

remove_index('synthetic_data/D2_TVAE.csv', 'synthetic_data/D2_TVAE.csv')