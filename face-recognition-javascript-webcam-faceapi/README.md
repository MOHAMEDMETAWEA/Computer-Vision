# AI Face Recognition Web Interface

A modern, real-time facial recognition application built with `face-api.js` and Javascript. This project allows for high-accuracy face detection, landmark identification, and face matching directly in the browser.

## 🚀 Features
- **Real-time Face Detection**: High-performance detection using SSD MobileNet V1.
- **Face Recognition**: Identify specific individuals using pre-labeled reference images.
- **Facial Landmarks**: 68-point facial landmark detection for feature extraction.
- **Modern UI**: Sleek, glassmorphism-inspired design with a dark mode aesthetic.
- **Robust Loading**: Integrated loading states and error handling for models and webcam access.

## 📁 Project Structure
- `weights/`: Contains the pre-trained neural network models (SSD MobileNet, Face Recognition, Landmarks).
- `labels/`: (User Action Required) Directory for storing reference images for recognition.
- `index.html`: The entry point with the modern UI structure.
- `script.js`: Main application logic including hardware acceleration and model management.
- `style.css`: Premium styling with CSS variables and responsive design.

## 🛠️ Setup Instructions

### 1. Recognition Setup (Optional)
To identify specific people, you need to populate the `labels` folder:
1. Create a subfolder for each person in `/labels/` (e.g., `/labels/John/`).
2. Add clear face images named `1.png` and `2.png` into each person's folder.
3. Update the `labels` array in `script.js` to match your folder names.

### 2. Running the Project
Since this uses `face-api.js`, it must be served through a web server to allow model loading (browsers block file system access for security).
- **VS Code**: Use the [Live Server](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) extension.
- **Node.js**: Use `npx serve` or `http-server`.
- **Python**: Use `python -m http.server`.

## 🧠 Neural Networks Used
- **SSD MobileNet V1**: For robust face detection.
- **Face Landmark 68**: For identifying facial features.
- **Face Recognition**: For generating face descriptors to match against reference images.

## 📄 License
This project utilizes the `face-api.js` library. Detailed model weights and library documentation can be found in the `face-api.js-master` directory.
