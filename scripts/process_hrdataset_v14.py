from __future__ import annotations

import csv
from pathlib import Path


INPUT_PATH = Path("data/HRDataset_v14.csv")
OUTPUT_PATH = Path("data/HRDataset_v14_no_gender_marital_ids.csv")
DROP_COLUMNS = {"GenderID", "MaritalStatusID"}


def main() -> None:
    with INPUT_PATH.open("r", newline="", encoding="utf-8-sig") as infile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            raise ValueError(f"No header row found in {INPUT_PATH}")

        missing = DROP_COLUMNS.difference(reader.fieldnames)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Missing expected columns: {missing_list}")

        fieldnames = [name for name in reader.fieldnames if name not in DROP_COLUMNS]

        with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                writer.writerow({name: row[name] for name in fieldnames})


if __name__ == "__main__":
    main()
