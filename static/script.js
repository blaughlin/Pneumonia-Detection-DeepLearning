function uploadImage() {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select an image file.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = `Prediction: ${data.Prediction} | Confidence: ${data.Confidence.toFixed(2)}`;
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("result").innerText = "Error: Unable to get prediction.";
    });
}

function selectExample(imagePath) {
    const formData = new FormData();
    formData.append("image_path", imagePath);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = `Prediction: ${data.Prediction} | Confidence: ${data.Confidence.toFixed(2)}`;
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("result").innerText = "Error: Unable to get prediction.";
    });
}