// DOM Elements
const imageInput = document.getElementById('imageInput');
const uploadArea = document.getElementById('uploadArea');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const detectBtn = document.getElementById('detectBtn');
const loading = document.getElementById('loading');
const resultCard = document.getElementById('resultCard');
const resultImage = document.getElementById('resultImage');
const summaryCard = document.getElementById('summaryCard');
const summaryContent = document.getElementById('summaryContent');
const stats = document.getElementById('stats');
const downloadBtn = document.getElementById('downloadBtn');
const confThreshold = document.getElementById('confThreshold');
const iouThreshold = document.getElementById('iouThreshold');
const confValue = document.getElementById('confValue');
const iouValue = document.getElementById('iouValue');
const featureVizCard = document.getElementById('featureVizCard');
const featureViz = document.getElementById('featureViz');
const architectureDiagram = document.getElementById('architectureDiagram');

// State
let currentImageFile = null;
let currentResultUrl = null;

// API Base URL
const API_BASE_URL = 'http://localhost:5000';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadArchitecture();
});

function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => {
        imageInput.click();
    });
    
    // File input change
    imageInput.addEventListener('change', handleImageUpload);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#764ba2';
    });
    
    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#667eea';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#667eea';
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFile(file);
        }
    });
    
    // Detect button
    detectBtn.addEventListener('click', detectObjects);
    
    // Download button
    downloadBtn.addEventListener('click', downloadResult);
    
    // Sliders
    confThreshold.addEventListener('input', (e) => {
        confValue.textContent = e.target.value;
    });
    
    iouThreshold.addEventListener('input', (e) => {
        iouValue.textContent = e.target.value;
    });
}

function handleImageUpload(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }
    
    currentImageFile = file;
    
    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.style.display = 'block';
    };
    reader.readAsDataURL(file);
    
    // Enable detect button
    detectBtn.disabled = false;
    
    // Hide previous results
    resultCard.style.display = 'none';
    summaryCard.style.display = 'none';
    featureVizCard.style.display = 'none';
}

async function detectObjects() {
    if (!currentImageFile) return;
    
    // Show loading
    loading.style.display = 'block';
    detectBtn.disabled = true;
    
    // Prepare form data
    const formData = new FormData();
    formData.append('image', currentImageFile);
    formData.append('confidence_threshold', confThreshold.value);
    formData.append('iou_threshold', iouThreshold.value);
    
    try {
        const response = await fetch(`${API_BASE_URL}/detect`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            alert('Error: ' + (data.error || 'Detection failed'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to connect to server. Make sure the backend is running.');
    } finally {
        loading.style.display = 'none';
        detectBtn.disabled = false;
    }
}

function displayResults(data) {
    // Display result image
    currentResultUrl = `${API_BASE_URL}${data.result_image_url}`;
    resultImage.src = currentResultUrl;
    resultCard.style.display = 'block';
    
    // Display summary
    summaryContent.innerHTML = `
        <div class="object-list">
            ${data.detections.map(detection => `
                <div class="object-item">
                    <span class="object-label">🎯 ${detection.label}</span>
                    <span class="object-confidence">${(detection.confidence * 100).toFixed(1)}%</span>
                </div>
            `).join('')}
            ${data.detections.length === 0 ? '<p>No objects detected</p>' : ''}
        </div>
    `;
    
    stats.innerHTML = `
        <div class="stats-content">
            <div class="total">Total Objects: ${data.total_objects}</div>
            <div class="params">
                <small>Confidence: ${data.parameters.confidence_threshold} | IOU: ${data.parameters.iou_threshold}</small>
            </div>
        </div>
    `;
    summaryCard.style.display = 'block';
    
    // Display feature visualization
    if (data.feature_viz) {
        featureViz.innerHTML = `<img src="${data.feature_viz}" alt="Feature Visualization">`;
        featureVizCard.style.display = 'block';
    }
    
    // Scroll to results
    resultCard.scrollIntoView({ behavior: 'smooth' });
}

function downloadResult() {
    if (currentResultUrl) {
        const link = document.createElement('a');
        link.href = currentResultUrl;
        link.download = 'detection_result.jpg';
        link.click();
    }
}

async function loadArchitecture() {
    try {
        const response = await fetch(`${API_BASE_URL}/architecture`);
        const data = await response.json();
        if (data.architecture_image) {
            architectureDiagram.innerHTML = `<img src="${data.architecture_image}" alt="YOLOv7 Architecture">`;
        }
    } catch (error) {
        console.error('Failed to load architecture:', error);
        architectureDiagram.innerHTML = '<p>Architecture diagram unavailable</p>';
    }
}