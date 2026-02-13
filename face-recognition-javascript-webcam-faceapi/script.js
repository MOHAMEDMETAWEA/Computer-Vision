const video = document.getElementById("video");
const loader = document.getElementById("loader");
const statusText = document.querySelector(".status-text");
const recognitionStatus = document.getElementById("recognition-status");

// Load models from the weights folder
async function loadModels() {
  statusText.innerText = "Loading Neural Networks...";
  try {
    await Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromUri("./weights"),
      faceapi.nets.faceRecognitionNet.loadFromUri("./weights"),
      faceapi.nets.faceLandmark68Net.loadFromUri("./weights"),
    ]);
    statusText.innerText = "Starting Camera...";
    startWebcam();
  } catch (error) {
    console.error("Error loading models:", error);
    statusText.innerText = "Failed to load models. Check weights folder.";
    statusText.style.color = "#ff4d4d";
  }
}

function startWebcam() {
  navigator.mediaDevices
    .getUserMedia({
      video: { width: 600, height: 450, frameRate: { ideal: 30 } },
      audio: false,
    })
    .then((stream) => {
      video.srcObject = stream;
      loader.classList.add("hidden");
    })
    .catch((error) => {
      console.error("Camera access denied:", error);
      statusText.innerText = "Please allow camera access to continue.";
      statusText.style.color = "#ff4d4d";
    });
}

/**
 * Loads images from the /labels folder to create reference descriptors.
 * Note: Folders must match labels and images should be named 1.png, 2.png, etc.
 */
async function getLabeledFaceDescriptions() {
  const labels = ["Mohamed", "CR7", "Messi"]; // Update these to match your labels folder
  const labeledDescriptors = [];

  for (const label of labels) {
    const descriptions = [];
    try {
      for (let i = 1; i <= 2; i++) {
        const imgPath = `./labels/${label}/${i}.jpeg`;
        const img = await faceapi.fetchImage(imgPath);
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();

        if (detections) {
          descriptions.push(detections.descriptor);
        }
      }

      if (descriptions.length > 0) {
        labeledDescriptors.push(new faceapi.LabeledFaceDescriptors(label, descriptions));
        console.log(`Loaded ${label} successfully`);
      }
    } catch (e) {
      console.warn(`Could not load images for ${label}. Skipping.`);
    }
  }
  return labeledDescriptors;
}

video.addEventListener("play", async () => {
  recognitionStatus.innerText = "Analyzing...";

  let labeledFaceDescriptors = await getLabeledFaceDescriptions();
  let faceMatcher = null;

  if (labeledFaceDescriptors.length > 0) {
    faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
    recognitionStatus.innerText = "Recognition Active";
  } else {
    recognitionStatus.innerText = "Detection Only";
    console.warn("No labels found in /labels folder. Running in detection mode.");
  }

  const canvas = faceapi.createCanvasFromMedia(video);
  canvas.style.position = 'absolute';
  canvas.style.top = '0';
  canvas.style.left = '0';
  canvas.style.zIndex = '10';
  canvas.style.pointerEvents = 'none';
  document.getElementById("video-wrapper").append(canvas);

  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);

  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video)
      .withFaceLandmarks()
      .withFaceDescriptors();

    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    resizedDetections.forEach((detection) => {
      const box = detection.detection.box;
      let label = "Unknown";

      if (faceMatcher) {
        const result = faceMatcher.findBestMatch(detection.descriptor);
        label = result.toString();
      }

      const drawBox = new faceapi.draw.DrawBox(box, {
        label: label,
        boxColor: "#00f2fe",
        drawLabelOptions: {
          fontColor: "#ffffff",
          fontSize: 16,
          padding: 8,
          background: "rgba(15, 23, 42, 0.8)"
        }
      });
      drawBox.draw(canvas);

      // Draw matching landmarks for extra visual flair
      faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);
    });
  }, 100);
});

// Start the process
loadModels();
