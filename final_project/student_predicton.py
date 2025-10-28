from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from student_pipeline import student_data_prep
import traceback
import os

# 🎓 FastAPI Uygulaması
app = FastAPI(title="Student Final Grade Prediction API")

# 🧠 Modeli yükle
try:
    model = joblib.load("voting_reg1.pkl")
    print("✅ Model yüklendi: voting_reg1.pkl")
except Exception as e:
    print(f"❌ Model yüklenemedi: {e}")
    model = None

# 📂 Eğitimde kullanılan kolonları yükle
try:
    training_columns = joblib.load("training_columns.pkl")
    print(f"✅ Loaded {len(training_columns)} training columns")
except FileNotFoundError:
    print("⚠️ Warning: training_columns.pkl not found. Run save_training_columns.py first!")
    training_columns = None


# 📋 Girdi Modeli
class StudentInput(BaseModel):
    school: str
    sex: str
    age: int
    address: str
    famsize: str
    Pstatus: str
    Medu: int
    Fedu: int
    Mjob: str
    Fjob: str
    reason: str
    guardian: str
    traveltime: int
    studytime: int
    failures: int
    schoolsup: str
    famsup: str
    paid: str
    activities: str
    nursery: str
    higher: str
    internet: str
    romantic: str
    famrel: int
    freetime: int
    goout: int
    Dalc: int
    Walc: int
    health: int
    absences: int
    G1: int
    G2: int


@app.get("/")
def home():
    return {"message": "🎓 Student Grade Prediction API is running!"}


@app.post("/predict")
def predict_grade(student: StudentInput):
    """
    Kullanıcıdan öğrenci bilgilerini alır, pipeline'dan geçirir, eksik kolonları tamamlar
    ve tahmin edilen G3 notunu döner.
    """
    try:
        # 1️⃣ JSON → DataFrame
        input_df = pd.DataFrame([student.dict()])
        print(f"\n📥 Input dataframe shape: {input_df.shape}")
        print(f"📥 Input columns: {input_df.columns.tolist()}")

        # 2️⃣ Pipeline preprocessing (G3 should NOT be present for prediction mode)
        try:
            X, _ = student_data_prep(input_df)  # This will auto-detect prediction mode
        except Exception as prep_err:
            print("❌ Hata preprocessing aşamasında:")
            traceback.print_exc()
            return {"error": f"Preprocessing failed: {str(prep_err)}"}

        print(f"🔧 After preprocessing: {X.shape[1]} columns")

        # 4️⃣ Eğitim kolonları ile hizalama
        if training_columns is not None:
            missing_cols = [col for col in training_columns if col not in X.columns]
            for col in missing_cols:
                X[col] = 0
            X = X[training_columns]
            print(f"✅ Columns aligned with training ({X.shape})")
        else:
            print("⚠️ training_columns.pkl bulunamadı, hizalama yapılmadı.")

        # 5️⃣ Tahmin
        if model is None:
            return {"error": "Model yüklenemedi, tahmin yapılamıyor."}

        prediction = model.predict(X)[0]
        print(f"📊 Prediction result: {prediction}")

        # 6️⃣ Sonuç döndür
        return {"predicted_G3": float(prediction)}

    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__} -> {e}")
        traceback.print_exc()
        return {"error": str(e)}
