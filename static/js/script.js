

document.getElementById('image-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);

    // Display the uploaded image
    const fileInput = document.getElementById('image-input');
    const file = fileInput.files[0];
    const previewImage = document.getElementById('uploaded-image');
    const reader = new FileReader();
    reader.onload = () => {
        previewImage.src = reader.result;
        previewImage.style.display = 'block';
    };
    reader.readAsDataURL(file);

    // Send the form data to the backend
    const response = await fetch('/predict_image', {
        method: 'POST',
        body: formData,
    });

    const result = await response.json();
    const resultDiv = document.getElementById('image-result');
    if (response.ok) {
        resultDiv.innerHTML = `
            <p><strong>Prediction:</strong> ${result.predicted_label}</p>
            <p><strong>Confidence:</strong></p>
            <ul>
                <li>Real: ${(result.confidence_scores[0] * 100).toFixed(2)}%</li>
                <li>Fake: ${(result.confidence_scores[1] * 100).toFixed(2)}%</li>
            </ul>
        `;
    } else {
        resultDiv.innerHTML = `<p>Error: ${result.error}</p>`;
    }
});

document.getElementById('video-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);

    // Display the uploaded video filename
    const fileInput = document.getElementById('video-input');
    const file = fileInput.files[0];
    const videoNameElement = document.getElementById('uploaded-video-name');
    videoNameElement.textContent = `Uploaded Video: ${file.name}`;
    videoNameElement.style.display = 'block';

    // Send the form data to the backend
    const response = await fetch('/predict_video', {
        method: 'POST',
        body: formData,
    });

    const result = await response.json();
    const resultDiv = document.getElementById('video-result');
    if (response.ok) {
        const predictions = result.predictions
            .map((label, index) => `<li>Frame ${index + 1}: ${label}</li>`)
            .join('');
        resultDiv.innerHTML = `
            <p><strong>Frame Predictions:</strong></p>
            <ul>${predictions}</ul>
        `;
    } else {
        resultDiv.innerHTML = `<p>Error: ${result.error}</p>`;
    }
});

document.getElementById('stream-video-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);

    // Send the video file to the backend for streaming
    const response = await fetch('/stream_video', {
        method: 'POST',
        body: formData,
    });

    // Display the streamed video
    const videoStream = document.getElementById('video-stream');
    videoStream.src = URL.createObjectURL(await response.blob());
    videoStream.style.display = 'block';
});




