# Student Grade Prediction Frontend

A modern, responsive web interface for the Student Final Grade Prediction API built with HTML, CSS, and JavaScript.

## 📋 Overview

This frontend provides a user-friendly form to input student data and receive predicted final grades (G3) from the FastAPI backend.

### Features

- 🎨 **Modern UI Design** - Beautiful gradient backgrounds and smooth animations
- 📱 **Fully Responsive** - Works perfectly on desktop, tablet, and mobile devices
- ⚡ **Real-time Feedback** - Range sliders with live value updates
- 🔄 **Easy Form Management** - Organized into logical sections with fieldsets
- 💬 **Clear Results** - Color-coded grade predictions with meaningful comments
- 🛡️ **Error Handling** - User-friendly error messages and retry options
- ⌨️ **Keyboard Shortcuts** - Alt+S to submit, Alt+R to reset, Alt+C to close errors
- 🔧 **API Integration** - Seamless connection to the FastAPI backend

## 📁 Files

- **index.html** - Main HTML structure with all form fields
- **styles.css** - Complete styling with responsive design and animations
- **script.js** - Form handling, API communication, and result display

## 🚀 Getting Started

### Prerequisites

1. The FastAPI backend should be running on `http://127.0.0.1:8000`
2. A modern web browser (Chrome, Firefox, Safari, Edge)

### Backend Setup

Make sure the FastAPI server is running:

```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

The server should start on `http://127.0.0.1:8000`

### Frontend Setup

1. Open the `index.html` file directly in a web browser, or
2. Serve it using a local HTTP server:

**Using Python 3:**
```bash
cd frontend
python -m http.server 8080
```

Then open `http://localhost:8080` in your browser

**Using Node.js (http-server):**
```bash
cd frontend
npx http-server
```

## 📝 Form Sections

### 📋 Personal Information
- School (Gabriel Pereira or Mousinho da Silveira)
- Sex (Male/Female)
- Age (15-25)
- Address Type (Urban/Rural)

### 👨‍👩‍👧‍👦 Family Information
- Family Size
- Parents Cohabitation Status
- Mother's & Father's Education Level
- Mother's & Father's Job
- Guardian

### 📚 Academic Information
- Reason to Choose School
- Travel Time
- Study Time per Week
- Number of Past Failures
- First Period Grade (G1)
- Second Period Grade (G2)
- Number of Absences

### 🎯 Support & Activities
- School Support
- Family Support
- Paid Classes
- Extracurricular Activities
- Nursery Attendance
- Desire for Higher Education
- Internet Access
- Romantic Relationship Status

### ❤️ Lifestyle & Health
- Family Relationships Quality (1-5 scale)
- Free Time After School (1-5 scale)
- Going Out with Friends (1-5 scale)
- Workday Alcohol Consumption (1-5 scale)
- Weekend Alcohol Consumption (1-5 scale)
- Current Health Status (1-5 scale)

## 🔌 API Integration

The frontend communicates with the backend API at `http://127.0.0.1:8000/predict`

### Request Format

```json
{
  "school": "GP",
  "sex": "M",
  "age": 17,
  "address": "U",
  "famsize": "GT3",
  "Pstatus": "T",
  "Medu": 4,
  "Fedu": 4,
  "Mjob": "teacher",
  "Fjob": "other",
  "reason": "course",
  "guardian": "mother",
  "traveltime": 1,
  "studytime": 3,
  "failures": 0,
  "schoolsup": "yes",
  "famsup": "yes",
  "paid": "no",
  "activities": "yes",
  "nursery": "yes",
  "higher": "yes",
  "internet": "yes",
  "romantic": "no",
  "famrel": 4,
  "freetime": 3,
  "goout": 4,
  "Dalc": 1,
  "Walc": 1,
  "health": 5,
  "absences": 2,
  "G1": 18,
  "G2": 17
}
```

### Response Format

**Success:**
```json
{
  "predicted_G3": 17.5
}
```

**Error:**
```json
{
  "error": "Error message describing what went wrong"
}
```

## 🎨 Design Details

### Color Scheme
- **Primary**: #3498db (Blue)
- **Secondary**: #2c3e50 (Dark Gray)
- **Success**: #27ae60 (Green)
- **Warning**: #f39c12 (Orange)
- **Danger**: #e74c3c (Red)

### Grade Indicators
- 🌟 **Excellent** (≥18)
- ✨ **Very Good** (≥16)
- 👍 **Good** (≥14)
- 👌 **Fair** (≥12)
- 📚 **Below Average** (≥10)
- ⚠️ **Poor** (<10)

## ⌨️ Keyboard Shortcuts

- **Alt + S** - Submit the form
- **Alt + R** - Reset the form
- **Alt + C** - Close error message
- **Escape** - Close error message

## 🔧 Browser Console

The frontend logs important information to the browser console:
- API connection test on page load
- Form data before sending
- API responses and errors
- Initialization messages

Open the browser DevTools (F12) and check the Console tab for debugging.

## 📱 Responsive Breakpoints

- **Desktop**: Full layout (>768px)
- **Tablet**: Optimized layout (481-768px)
- **Mobile**: Stacked layout (<480px)

## 🐛 Troubleshooting

### "Cannot connect to the server"
- Make sure the FastAPI backend is running on `http://127.0.0.1:8000`
- Check that port 8000 is not blocked by a firewall
- Verify no other service is running on port 8000

### Form won't submit
- Check that all required fields are filled
- Open browser console (F12) to see specific errors
- Make sure the backend is responding to requests

### Blank grade result
- Check the browser console for error messages
- Verify all form values are correct
- Ensure the model files (voting_reg1.pkl, training_columns.pkl) are in the backend directory

### CORS Errors
If you see CORS errors in the console, you may need to update the backend to include CORS support:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify ["http://localhost:8080"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📊 Testing

You can test the form with these sample values:

**High Performer:**
- G1: 18, G2: 17, Age: 16
- Study Time: 5-10 hours
- Failures: 0
- Higher Education: Yes
- School Support: Yes

**Low Performer:**
- G1: 10, G2: 8, Age: 18
- Study Time: <2 hours
- Failures: 2
- Higher Education: No
- School Support: No

## 🔐 Security Notes

- The frontend sends sensitive data to the backend; ensure HTTPS is used in production
- Never expose API keys or credentials in the frontend code
- Validate all user inputs on the backend

## 📜 License

This project is part of the Student Performance Prediction System.

## 👤 Author

Created for the Student Grade Prediction Project

## 🤝 Support

For issues or questions, please check:
1. Browser console (F12) for error messages
2. Backend logs for API errors
3. Ensure both frontend and backend are properly configured

---

**Last Updated**: October 2025
