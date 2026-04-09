import pandas as pd
from arfpy import arf
from sklearn.datasets import load_iris
import os

try:
    from .preprocessing import prepare_training_dataframe
except ImportError:
    from preprocessing import prepare_training_dataframe

def generate(train_data, n_generated, output_dir):
    # iris = load_iris()
    # print(iris['feature_names'])
    # df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

    df = prepare_training_dataframe(train_data)
    myarf = arf.arf(x = df)
    myarf.forde()
    new_data = myarf.forge(n = n_generated)
    float_cols = new_data.select_dtypes(include='float').columns
    new_data[float_cols] = new_data[float_cols].round(3)
    output_dir = os.path.join('synthetic_data', f'{output_dir}')
    new_data.to_csv(output_dir)
    return new_data

if __name__ == "__main__":
    generate('data/kaggle/ibm_hr.csv', 1500, 'kaggle_arf.csv')
