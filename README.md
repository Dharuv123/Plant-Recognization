# 🌿 Plant Recognition System

A full-stack web application that identifies plants from an uploaded image and returns their **botanical name, medicinal properties, and medicinal uses** — built to help users quickly learn about the medicinal value of plants around them.

## 🔍 Overview

Upload a photo of a plant, and the app uses a trained Convolutional Neural Network (CNN) to classify the species, then returns:
- **Botanical (scientific) name**
- **Medicinal properties** of the plant
- **Medicinal uses** — common applications in traditional/modern medicine

## ✨ Features

- 📸 Image upload and real-time plant classification
- 🧠 CNN-based image classification model (TensorFlow) achieving **95% classification accuracy**
- 🌱 Returns botanical name, medicinal properties, and uses for the identified plant
- ⚡ REST API backend (Flask) serving predictions to a React frontend
- 🖼️ Image preprocessing and augmentation pipeline for robust predictions across varied image conditions (lighting, angle, background)

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React |
| Backend | Flask (REST API) |
| ML Model | TensorFlow (CNN) |
| Database | MongoDB |
| Image Processing | TensorFlow / Keras preprocessing utilities |

## 📂 Project Structure

```
plant-recognition-system/
├── backend/
│   ├── app.py                # Flask app & API routes
│   ├── model/                # Trained CNN model files
│   ├── utils/                # Image preprocessing & augmentation helpers
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/       # React components (Upload, Result, etc.)
│   │   └── App.js
│   └── package.json
└── README.md
```

*(Update this structure to match your actual folder layout.)*

## ⚙️ How It Works

1. User uploads a plant image via the React frontend.
2. Image is sent to the Flask backend via a REST API call.
3. Backend preprocesses the image (resizing, normalization, augmentation-consistent transforms).
4. The trained CNN model classifies the plant species.
5. Backend looks up the predicted species in the medicinal-plant database (MongoDB) and retrieves the botanical name, medicinal properties, and uses.
6. Result is returned to the frontend and displayed to the user.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js & npm
- MongoDB instance (local or Atlas)

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

### Environment Variables
Create a `.env` file in the backend directory:
```
MONGO_URI=your_mongodb_connection_string
```

## 📊 Model Details

- **Architecture:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow
- **Accuracy:** 95% on the test dataset
- **Techniques used:** Image preprocessing, data augmentation (rotation, flipping, zoom, etc.) to improve generalization across varied real-world images

## 🔮 Future Improvements

- Add confidence score display alongside predictions
- Expand dataset to cover more plant species
- Add multilingual support for medicinal use descriptions
- Deploy live demo (Vercel/Render)

## 👤 Author

**Dharuvkumar Bhansali**
[LinkedIn](https://www.linkedin.com/in/dharuvkumar-bhansali-b1319425b/) · [GitHub](https://github.com/Dharuv123)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
