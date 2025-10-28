import os
import pandas as pd
import joblib
from student_pipeline import student_data_prep


def main():
    # CSV assumed to be in the repo root next to this script
    csv_path = os.path.join(os.path.dirname(__file__), "student-por.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found at {csv_path}")
        return

    print(f"Reading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    print("Running preprocessing in training mode to extract training columns...")
    X, y = student_data_prep(df, is_training=True)

    training_columns = list(X.columns)
    print(f"Found {len(training_columns)} training columns. Saving to training_columns.pkl")
    joblib.dump(training_columns, "training_columns.pkl")
    print("Saved training_columns.pkl")


if __name__ == "__main__":
    main()
