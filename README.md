# 🦠 Malaria Detection AI

Deep learning app for classifying microscopy cell images as **Parasitized** or **Uninfected** using a fine-tuned **ResNet50** model.

## Results
- Validation Accuracy: **92.8%**
- Input Size: **128x128**
- Transfer Learning with ResNet50
- Streamlit app for live predictions

## Project Highlights
- Improved performance by increasing image resolution from 64x64 to 128x128
- Used ResNet50 preprocessing for stronger feature extraction
- Evaluated model with confusion matrix, classification metrics, and training curves

## Tech Stack
- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Matplotlib
- Seaborn

## Run Locally
```bash
pip install -r model/requirements.txt
streamlit run model/app.py
