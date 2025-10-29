# ✅ Quick Setup Verification Checklist

## 🚀 Before Testing - Verify Everything is Ready

### 1️⃣ Backend Setup
- [ ] Navigate to backend folder: `cd c:\Users\merve\Desktop\final_project\backend`
- [ ] Check files exist:
  - [ ] `main.py` ✓
  - [ ] `voting_reg1.pkl` ✓ (required for predictions)
  - [ ] `training_columns.pkl` ✓ (required for predictions)
  - [ ] `student_pipeline.py` ✓
  - [ ] `requirements.txt` ✓

### 2️⃣ Backend Start
- [ ] Run: `python -m uvicorn main:app --reload`
- [ ] Check for this message:
  ```
  ✅ Model yüklendi: voting_reg1.pkl
  ✅ Loaded 30 training columns
  INFO:     Uvicorn running on http://127.0.0.1:8000
  ```

### 3️⃣ Frontend Setup
- [ ] Navigate to frontend: `c:\Users\merve\Desktop\final_project\frontend`
- [ ] Verify files exist:
  - [ ] `index.html` ✓
  - [ ] `styles.css` ✓
  - [ ] `script.js` ✓

### 4️⃣ Frontend Start (Choose One)
**Option A: Direct file**
- [ ] Double-click `index.html` to open in browser

**Option B: HTTP Server**
- [ ] In new PowerShell, run: `python -m http.server 8080`
- [ ] Open browser: `http://localhost:8080`

### 5️⃣ Test the Prediction

**In Browser:**
1. [ ] Form loads (you see all fields)
2. [ ] Fill ALL form fields (look for red borders - those are required)
3. [ ] Click "🚀 Predict Grade"
4. [ ] **Important:** Open browser console (F12) to see debug info

### 6️⃣ Check Console Output (F12)

When you click predict, look for these in console:

✅ **Success indicators:**
- `📤 Sending data to API:` - Data collected
- `📌 API URL: http://127.0.0.1:8000/predict` - Correct endpoint
- `📊 Number of fields: 32` - All fields sent
- `📥 Response Status: 200` - Success response
- `✅ Success! Displaying results...` - Results shown

❌ **Error indicators:**
- `⚠️ Cannot connect to API server` → Backend not running
- `📥 Response Status: 422` → Form data invalid
- `📥 Response Status: 500` → Backend error
- `❌ Fetch Error:` → Connection failed

---

## 🎯 Most Common Issues & Fixes

### Issue 1: "Cannot connect to the server"
**Cause:** Backend not running
**Fix:**
```powershell
# Stop current backend (Ctrl+C)
# Then:
cd c:\Users\merve\Desktop\final_project\backend
python -m uvicorn main:app --reload
```

### Issue 2: "Response Status: 422"
**Cause:** Invalid form data
**Fix:**
- Check console for validation error details
- Ensure ALL required fields are filled
- Check field values match expected format

### Issue 3: Results not showing (Status 200 but no results)
**Cause:** DOM elements not found
**Fix:**
- Check console says: `🎯 displayResults called with grade:`
- If not, something prevented function call
- Try: Clear cache (Ctrl+Shift+Delete) and reload

### Issue 4: Model files not found
**Cause:** Files missing from backend folder
**Fix:**
- Verify `voting_reg1.pkl` exists in backend folder
- Verify `training_columns.pkl` exists in backend folder
- If missing, model needs to be trained first

---

## 🔍 Detailed Diagnosis Steps

### Step 1: Verify Backend is Running
```powershell
# Open new terminal and try:
curl http://127.0.0.1:8000/
# Should return: {"message": "🎓 Student Grade Prediction API is running!"}
```

### Step 2: Verify Model Files
```powershell
cd c:\Users\merve\Desktop\final_project\backend
python -c "import joblib; joblib.load('voting_reg1.pkl'); print('✅ Model OK')"
python -c "import joblib; cols = joblib.load('training_columns.pkl'); print(f'✅ Columns OK: {len(cols)}')"
```

### Step 3: Verify Frontend Files
```powershell
cd c:\Users\merve\Desktop\final_project\frontend
# Check all 3 files exist:
dir /B
# Should show: index.html, styles.css, script.js, README.md
```

### Step 4: Test API Manually
- Open: `http://127.0.0.1:8000/docs`
- Click "Try it out" on the POST /predict endpoint
- Fill with sample data
- Click "Execute"
- Check response

---

## 📝 Sample Test Data (Fill These Values)

| Field | Value |
|-------|-------|
| school | GP |
| sex | M |
| age | 17 |
| address | U |
| famsize | GT3 |
| Pstatus | T |
| Medu | 4 |
| Fedu | 4 |
| Mjob | teacher |
| Fjob | other |
| reason | course |
| guardian | mother |
| traveltime | 1 |
| studytime | 3 |
| failures | 0 |
| schoolsup | yes |
| famsup | yes |
| paid | no |
| activities | yes |
| nursery | yes |
| higher | yes |
| internet | yes |
| romantic | no |
| famrel | 4 |
| freetime | 3 |
| goout | 4 |
| Dalc | 1 |
| Walc | 1 |
| health | 5 |
| absences | 2 |
| G1 | 18 |
| G2 | 17 |

Expected result: Around **17-18** (Excellent grade)

---

## ✨ When Everything Works

You should see:
1. Form displays ✓
2. Fill form ✓
3. Click predict ✓
4. Loading spinner shows ✓
5. **Results appear** ✓
   - Predicted grade shows (e.g., 17.5)
   - Grade category shows (e.g., "Very Good ✨")
   - Color-coded background appears
   - Meaningful comment shows

---

## 🚨 If Still Having Issues

1. **Copy full console output** (Ctrl+A, Ctrl+C)
2. **Screenshot the error** (if any red text)
3. **Check backend console** for errors
4. **Report these details:**
   - Exact error message
   - Browser console output
   - Backend console output
   - Are model files present?
   - Is backend running?

---

## 📞 Need Help?

1. **Check:** `DEBUGGING_GUIDE.md` (detailed debugging steps)
2. **Check:** `QUICK_START.md` (setup instructions)
3. **Check:** Backend console for errors
4. **Share:** Console output & error details

---

**Ready to test? Follow the checklist above! ✅**
