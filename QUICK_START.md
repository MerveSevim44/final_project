# Quick Start Guide

## 🚀 Running the Application

### Step 1: Start the Backend

Open a PowerShell terminal and navigate to the backend folder:

```powershell
cd c:\Users\merve\Desktop\final_project\backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
✅ Model yüklendi: voting_reg1.pkl
✅ Loaded 30 training columns
```

### Step 2: Open the Frontend

**Option A: Direct File Access**
- Navigate to `c:\Users\merve\Desktop\final_project\frontend`
- Double-click `index.html` to open it in your default browser

**Option B: Using Python HTTP Server** (Recommended)
Open another PowerShell terminal:

```powershell
cd c:\Users\merve\Desktop\final_project\frontend
python -m http.server 8080
```

Then open your browser and go to: `http://localhost:8080`

**Option C: Using VS Code Live Server**
- Install the "Live Server" extension in VS Code
- Right-click on `index.html`
- Select "Open with Live Server"

### Step 3: Use the Application

1. Fill in all the form fields
2. Click "🚀 Predict Grade" button
3. View your predicted final grade

## 📋 Form Input Guide

### Quick Fill Example

Try these values for a test:

| Field | Value |
|-------|-------|
| School | Gabriel Pereira (GP) |
| Sex | Male |
| Age | 17 |
| Address | Urban |
| Family Size | Greater than 3 |
| Parents Status | Together |
| Mother's Education | Higher Education |
| Father's Education | Higher Education |
| Study Time | 5-10 hours |
| Failures | 0 |
| G1 (First Period) | 18 |
| G2 (Second Period) | 17 |
| School Support | Yes |
| Family Support | Yes |
| Higher Education | Yes |

## 🔍 Monitoring

### Check Backend Status
- Open browser console (F12)
- Look for "✅ API connection successful"
- Check Network tab to see API requests

### Check API Response
- Open browser DevTools (F12)
- Go to Network tab
- Click "Predict Grade"
- Click the request to "predict"
- Check the Response tab for the returned grade

## 📊 Understanding Results

- **Excellent (≥18)** 🌟: Outstanding performance
- **Very Good (≥16)** ✨: Great job
- **Good (≥14)** 👍: On the right track
- **Fair (≥12)** 👌: Decent performance
- **Below Average (≥10)** 📚: Needs improvement
- **Poor (<10)** ⚠️: Requires significant help

## ⚠️ Troubleshooting

### "Cannot connect to the server" Error

**Solution:**
1. Check if backend is running (should see Uvicorn running message)
2. Verify you're not running on port 8000 elsewhere:
```powershell
netstat -ano | findstr :8000
```
3. If port is in use, stop that process or change the port:
```powershell
python -m uvicorn main:app --reload --port 8001
```

### Form Won't Submit

**Solution:**
1. Make sure all fields are filled (red borders show required fields)
2. Open browser console (F12) and check for errors
3. Verify backend is responding:
```powershell
# In a new PowerShell terminal
curl http://127.0.0.1:8000/
```

### Blank Result Page

**Solution:**
1. Check that model files exist in backend folder:
   - `voting_reg1.pkl`
   - `training_columns.pkl`
2. Check backend console for error messages
3. Try a different set of input values

## 💡 Tips & Tricks

### Keyboard Shortcuts
- **Alt + S**: Quick submit form
- **Alt + R**: Quick reset form  
- **Escape**: Close error messages

### Form Navigation
- Use **Tab** to move between fields
- Use **Shift + Tab** to go back
- Range sliders update values in real-time

### Testing Different Scenarios

**Low Grade Student:**
- G1: 8, G2: 10
- Study Time: <2 hours
- Failures: 2+

**High Grade Student:**
- G1: 18, G2: 19
- Study Time: 5-10 hours
- Failures: 0

## 🔗 Important URLs

| Component | URL |
|-----------|-----|
| Backend API | http://127.0.0.1:8000 |
| Frontend (Local) | http://localhost:8080 |
| API Prediction | http://127.0.0.1:8000/predict |
| API Docs | http://127.0.0.1:8000/docs |

## 📝 Backend API Documentation

While the backend is running, you can access interactive API documentation:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## 🛑 Stopping the Application

**To stop the backend:**
- Press `Ctrl+C` in the backend PowerShell terminal

**To stop the frontend server:**
- Press `Ctrl+C` in the frontend PowerShell terminal

## 📂 Project Structure

```
final_project/
├── backend/
│   ├── main.py              (FastAPI application)
│   ├── requirements.txt     (Python dependencies)
│   ├── voting_reg1.pkl      (Trained model)
│   ├── training_columns.pkl (Feature mapping)
│   └── student_pipeline.py  (Data preprocessing)
└── frontend/
    ├── index.html           (Main HTML form)
    ├── styles.css           (Styling)
    ├── script.js            (Form handling & API calls)
    └── README.md            (Documentation)
```

## 🎓 Next Steps

1. Test with various inputs to understand the model
2. Explore the API documentation at `/docs`
3. Check the model accuracy with known student data
4. Customize styling or form fields as needed

---

**Ready to predict? Open the frontend and start filling the form!** 🚀
