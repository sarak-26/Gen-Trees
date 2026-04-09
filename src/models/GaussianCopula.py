import os

try:
    from sdv.metadata import Metadata
    from sdv.single_table import GaussianCopulaSynthesizer
except ModuleNotFoundError as exc:
    Metadata = None
    GaussianCopulaSynthesizer = None
    _SDV_IMPORT_ERROR = exc
else:
    _SDV_IMPORT_ERROR = None

try:
    from .preprocessing import prepare_training_dataframe
except ImportError:
    from preprocessing import prepare_training_dataframe


def generate(train_data, n_generated, output_dir):
    if GaussianCopulaSynthesizer is None or Metadata is None:
        raise ModuleNotFoundError(
            "GaussianCopula generation requires the 'sdv' package to be installed."
        ) from _SDV_IMPORT_ERROR

    df = prepare_training_dataframe(train_data)

    metadata = Metadata.detect_from_dataframe(df)
    model = GaussianCopulaSynthesizer(metadata)
    model.fit(df)
    new_data = model.sample(n_generated)
    float_cols = new_data.select_dtypes(include="float").columns
    new_data[float_cols] = new_data[float_cols].round(3)

    os.makedirs("synthetic_data", exist_ok=True)
    output_path = os.path.join("synthetic_data", f"{output_dir}")
    new_data.to_csv(output_path)
    return new_data


if __name__ == "__main__":
    generate("data/kaggle/ibm_hr.csv", 1500, "kaggle_GaussianCopula.csv")
