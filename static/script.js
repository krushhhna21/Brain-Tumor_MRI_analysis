// ==================== DOM ELEMENTS ====================
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const resultsSection = document.getElementById('resultsSection');
const emptyState = document.getElementById('emptyState');
const previewImg = document.getElementById('previewImg');
const spinner = document.getElementById('spinner');
const resultContent = document.getElementById('resultContent');
const className = document.getElementById('className');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceLevel = document.getElementById('confidenceLevel');
const confidenceFill = document.getElementById('confidenceFill');

// ==================== DRAG & DROP ====================
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// ==================== FILE INPUT ====================
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// ==================== HANDLE FILE SELECTION ====================
async function handleFileSelect(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        return;
    }

    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size must be less than 16MB');
        return;
    }

    // Show loading state
    emptyState.style.display = 'none';
    resultsSection.style.display = 'grid';
    spinner.style.display = 'block';
    resultContent.style.display = 'none';

    // Display preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Send to server
    await sendPredictionRequest(file);
}

// ==================== SEND PREDICTION REQUEST ====================
async function sendPredictionRequest(file) {
    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (error) {
        showError('Error uploading image: ' + error.message);
    }
}

// ==================== DISPLAY RESULTS ====================
function displayResults(data) {
    // Hide spinner
    spinner.style.display = 'none';
    resultContent.style.display = 'block';

    // Update class name
    className.textContent = data.class;

    // Update confidence with animation
    const confidence = data.confidence;
    confidenceValue.textContent = confidence.toFixed(1);
    confidenceLevel.textContent = `${data.confidence_level} Confidence`;
    
    // Animate confidence bar
    confidenceFill.style.setProperty('--fill-percent', confidence + '%');
    confidenceFill.style.width = confidence + '%';

    // Set confidence color based on level
    updateConfidenceColor(data.confidence_color);
}

// ==================== UPDATE CONFIDENCE COLOR ====================
function updateConfidenceColor(colorClass) {
    confidenceLevel.className = 'confidence-level';
    
    if (colorClass === 'success') {
        confidenceLevel.style.color = '#2ECC71';
    } else if (colorClass === 'warning') {
        confidenceLevel.style.color = '#F39C12';
    } else {
        confidenceLevel.style.color = '#3498DB';
    }
}

// ==================== RESET UPLOAD ====================
function resetUpload() {
    // Reset form
    fileInput.value = '';
    uploadArea.classList.remove('dragover');
    
    // Hide results
    resultsSection.style.display = 'none';
    emptyState.style.display = 'block';
    
    // Reset content
    previewImg.src = '';
    className.textContent = 'Analyzing...';
    confidenceValue.textContent = '0';
    confidenceFill.style.width = '0%';
    confidenceLevel.textContent = 'Analyzing';
}

// ==================== SHOW ERROR ====================
function showError(message) {
    alert('Error: ' + message);
    resetUpload();
}

// ==================== KEYBOARD SUPPORT ====================
document.addEventListener('keydown', (e) => {
    // Open file dialog on Ctrl+O
    if (e.ctrlKey && e.key === 'o') {
        e.preventDefault();
        fileInput.click();
    }
});

// ==================== COPY TO CLIPBOARD ====================
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        alert('Copied to clipboard!');
    });
}

// ==================== INITIAL LOAD ====================
window.addEventListener('load', () => {
    console.log('🧠 Brain Tumor MRI Classifier loaded successfully!');
});
