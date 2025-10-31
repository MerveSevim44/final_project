// ===========================
// Configuration
// ===========================

const API_URL = 'http://127.0.0.1:8000/predict';
const FORM_ID = 'predictionForm';
const RESULTS_CONTAINER_ID = 'resultsContainer';
const ERROR_CONTAINER_ID = 'errorContainer';
const LOADING_CONTAINER_ID = 'loadingContainer';

// ===========================
// DOM Elements
// ===========================

const form = document.getElementById(FORM_ID);
const resultsContainer = document.getElementById(RESULTS_CONTAINER_ID);
const errorContainer = document.getElementById(ERROR_CONTAINER_ID);
const loadingContainer = document.getElementById(LOADING_CONTAINER_ID);

// ===========================
// Range Slider Value Display
// ===========================

document.addEventListener('DOMContentLoaded', () => {
    // Setup range sliders
    const rangeInputs = [
        { id: 'famrel', displayId: 'famrelValue' },
        { id: 'freetime', displayId: 'freetimeValue' },
        { id: 'goout', displayId: 'gooutValue' },
        { id: 'Dalc', displayId: 'DalcValue' },
        { id: 'Walc', displayId: 'WalcValue' },
        { id: 'health', displayId: 'healthValue' }
    ];

    rangeInputs.forEach(({ id, displayId }) => {
        const input = document.getElementById(id);
        const display = document.getElementById(displayId);
        
        if (input && display) {
            input.addEventListener('input', (e) => {
                display.textContent = e.target.value;
            });
        }
    });

    // Form submission
    form.addEventListener('submit', handleFormSubmit);
    // Autofill test data button
    const autofillBtn = document.getElementById('autofillTestBtn');
    if (autofillBtn) {
        autofillBtn.addEventListener('click', autofillTestData);
    }
});

// ===========================
// Form Submission Handler
// ===========================

async function handleFormSubmit(e) {
    e.preventDefault();
    
    // Hide previous results/errors
    resultsContainer.classList.add('hidden');
    errorContainer.classList.add('hidden');
    
    // Show loading spinner
    showLoading();

    try {
        console.log('🚀 Form submission initiated');
        // Collect form data
        const formData = collectFormData();
        
        console.log('📤 Sending data to API:', formData);
        console.log('📌 API URL:', API_URL);
        console.log('📊 Number of fields:', Object.keys(formData).length);



        // Send request to backend
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        console.log('📥 Response Status:', response.status);
        console.log('📥 Response OK:', response.ok);

        const data = await response.json();
        console.log('📥 API Response:', data);
        console.log('📥 predicted_G3 value:', data.predicted_G3);

        // Hide loading spinner
        hideLoading();

        // Handle response
        if (response.ok && data.predicted_G3 !== undefined) {
            console.log('✅ Success! Displaying results...');
            displayResults(data.predicted_G3);
        } else if (data.error) {
            console.error('❌ Backend returned error:', data.error);
            displayError(data.error);
        } else {
            console.error('❌ Unexpected response format:', data);
            displayError('Unexpected response from server. Please try again.');
        }

    } catch (error) {
        console.error('❌ Fetch Error:', error);
        console.error('❌ Error message:', error.message);
        console.error('❌ Error stack:', error.stack);
        hideLoading();
        displayError(`Failed to connect to the server: ${error.message}`);
    }
}

// ===========================
// Collect Form Data
// ===========================

function collectFormData() {
    const formElements = form.elements;
    const data = {};

    for (let element of formElements) {
        if (element.name && element.type !== 'submit' && element.type !== 'reset') {
            const value = element.value;
            
            // Convert to appropriate type
            if (element.type === 'number' || element.type === 'range') {
                data[element.name] = parseInt(value, 10);
            } else {
                data[element.name] = value;
            }
        }
    }

    return data;
}

// ===========================
// Display Results
// ===========================

function displayResults(grade) {
    console.log('🎯 displayResults called with grade:', grade);
    
    const gradeIndicator = document.getElementById('gradeIndicator');
    const gradeComment = document.getElementById('gradeComment');
    const predictionResult = document.getElementById('predictionResult');
    
    if (!gradeIndicator || !gradeComment || !predictionResult) {
        console.error('❌ Missing DOM elements for results display');
        displayError('Error displaying results - missing UI elements');
        return;
    }
    
    // Round grade to 2 decimal places
    const roundedGrade = Math.round(grade * 100) / 100;
    console.log('🔢 Rounded grade:', roundedGrade);
    
    predictionResult.textContent = roundedGrade;

    // Determine grade category and comment
    let gradeCategory = '';
    let comment = '';
    let emoji = '';

    if (grade >= 18) {
        gradeCategory = 'Excellent';
        emoji = '🌟';
        comment = 'Outstanding performance! Keep up the excellent work!';
    } else if (grade >= 16) {
        gradeCategory = 'Very Good';
        emoji = '✨';
        comment = 'Great job! You are performing very well.';
    } else if (grade >= 14) {
        gradeCategory = 'Good';
        emoji = '👍';
        comment = 'Good performance! You are on the right track.';
    } else if (grade >= 12) {
        gradeCategory = 'Fair';
        emoji = '👌';
        comment = 'Fair performance. Consider focusing more on your studies.';
    } else if (grade >= 10) {
        gradeCategory = 'Below Average';
        emoji = '📚';
        comment = 'Below average performance. Seek additional help and focus on improvement.';
    } else {
        gradeCategory = 'Poor';
        emoji = '⚠️';
        comment = 'Poor performance. Please seek tutoring and increase study time.';
    }

    console.log('📊 Grade category:', gradeCategory, 'Emoji:', emoji);
    
    gradeIndicator.textContent = emoji;
    gradeIndicator.title = gradeCategory;
    gradeComment.textContent = `${gradeCategory}: ${comment}`;

    // Update results container background based on grade
    const resultsCard = resultsContainer.querySelector('.results-card');
    if (!resultsCard) {
        console.error('❌ Missing results card element');
        displayError('Error displaying results - missing card element');
        return;
    }
    
    if (grade >= 16) {
        resultsCard.style.background = 'linear-gradient(135deg, #27ae60 0%, #229954 100%)';
    } else if (grade >= 12) {
        resultsCard.style.background = 'linear-gradient(135deg, #f39c12 0%, #d68910 100%)';
    } else {
        resultsCard.style.background = 'linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)';
    }

    // Show results container
    console.log('🎉 Showing results container...');
    resultsContainer.classList.remove('hidden');
    
    console.log('✅ Results displayed successfully!');
    scrollToResults();
}

// ===========================
// Display Error
// ===========================

function displayError(message) {
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = message;
    errorContainer.classList.remove('hidden');
    scrollToError();
}

// ===========================
// UI Control Functions
// ===========================

function showLoading() {
    loadingContainer.classList.remove('hidden');
}

function hideLoading() {
    loadingContainer.classList.add('hidden');
}

function closeError() {
    errorContainer.classList.add('hidden');
}

function scrollToForm() {
    form.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function scrollToResults() {
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function scrollToError() {
    errorContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ===========================
// Utility Functions
// ===========================



// ===========================
// Keyboard Shortcuts
// ===========================

-
// ===========================
// API Connection Test
// ===========================

// Test API connection on page load
document.addEventListener('DOMContentLoaded', () => {
    testAPIConnection();
});

async function testAPIConnection() {
    try {
        const response = await fetch('http://127.0.0.1:8000/', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (response.ok) {
            console.log('✅ API connection successful');
        } else {
            console.warn('⚠️ API responded with status:', response.status);
        }
    } catch (error) {
        console.warn('⚠️ Cannot connect to API server at http://127.0.0.1:8000');
        console.warn('Make sure the FastAPI server is running with: uvicorn main:app --reload');
    }
}

// ===========================
// Autofill Test Data
// ===========================

const TEST_DATA = {
    school: 'GP',
    sex: 'F',
    age: 17,
    address: 'U',
    famsize: 'GT3',
    Pstatus: 'T',
    Medu: 4,
    Fedu: 4,
    Mjob: 'teacher',
    Fjob: 'services',
    reason: 'course',
    guardian: 'mother',
    traveltime: 1,
    studytime: 2,
    failures: 0,
    schoolsup: 'no',
    famsup: 'yes',
    paid: 'no',
    activities: 'yes',
    nursery: 'yes',
    higher: 'yes',
    internet: 'yes',
    romantic: 'no',
    famrel: 4,
    freetime: 3,
    goout: 2,
    Dalc: 1,
    Walc: 1,
    health: 3,
    absences: 2,
    G1: 14,
    G2: 15
};

function autofillTestData() {
    console.log('🧪 Autofill: filling form with test data');
    for (const [key, value] of Object.entries(TEST_DATA)) {
        // Prefer getElementById, fallback to querySelector by name
        let el = document.getElementById(key);
        if (!el) el = document.querySelector(`[name="${key}"]`);
        if (!el) continue;

        // Set value depending on element type
        if (el.tagName === 'SELECT' || el.type === 'select-one') {
            el.value = String(value);
            el.dispatchEvent(new Event('change', { bubbles: true }));
        } else if (el.type === 'range' || el.type === 'number' || el.type === 'text') {
            el.value = value;
            // For range inputs, also update display spans
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
        } else {
            try { el.value = value; } catch (e) { /* ignore */ }
        }

        // If the field has a paired display <span id="<name>Value">, update it
        const displaySpan = document.getElementById(`${key}Value`);
        if (displaySpan) displaySpan.textContent = String(value);
    }

    // Clear previous results/errors and hide them
    if (resultsContainer) resultsContainer.classList.add('hidden');
    if (errorContainer) errorContainer.classList.add('hidden');

    // Scroll to form so user can review
    setTimeout(() => {
        form.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 150);
}

// ===========================
// Form Validation (Optional Enhancement)
// ===========================

function validateForm() {
    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;

    requiredFields.forEach(field => {
        if (!field.value) {
            field.classList.add('error');
            isValid = false;
        } else {
            field.classList.remove('error');
        }
    });

    return isValid;
}

// Optional: Add real-time validation
form.addEventListener('change', (e) => {
    if (e.target.hasAttribute('required') && e.target.value) {
        e.target.classList.remove('error');
    }
});

// ===========================
// Browser Compatibility
// ===========================

// Polyfill for older browsers
if (!String.prototype.startsWith) {
    String.prototype.startsWith = function(search, pos) {
        pos = !pos ? 0 : +pos;
        return this.substr(pos, search.length) === search;
    };
}

// Log initialization
console.log('🎓 Student Grade Prediction Frontend initialized');
console.log('API Endpoint:', API_URL);
