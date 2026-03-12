from sdv.single_table import TVAESynthesizer
from sdv.metadata import Metadata
import pandas as pd
import os

def generate(train_data, n_generated, output_dir):

    df = pd.read_csv(train_data, header=0)

    metadata = Metadata.detect_from_dataframe(df)
    model = TVAESynthesizer(metadata)
    model.fit(df)
    new_data = model.sample(n_generated)
    float_cols = new_data.select_dtypes(include='float').columns
    new_data[float_cols] = new_data[float_cols].round(3)
    output_dir = os.path.join('synthetic_data', f'{output_dir}')
    new_data.to_csv(output_dir)
    return new_data

generate('data/kaggle/ibm_hr.csv', 1500, 'kaggle_TVAE.csv')