import joblib
import pandas as pd

df = pd.read_csv("student-por.csv")

random_user = df.sample(1, random_state=45)

new_model = joblib.load("voting_reg.pkl")

new_model.predict(random_user)

from student_pipeline import student_data_prep

X, y = student_data_prep(df)

random_user = X.sample(1, random_state=50)

new_model = joblib.load("voting_reg.pkl")

new_model.predict(random_user)