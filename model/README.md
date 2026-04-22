# Malaria Detection AI

This Streamlit app uses a fine-tuned ResNet50-based model to classify blood smear cell images as parasitized or uninfected. To reduce overconfident borderline classifications, the app includes an uncertainty zone for cases that should be manually reviewed. The final model achieved 97.88% validation accuracy with a validation loss of 0.0588.

## Features

- Upload custom cell images for analysis
- Try sample images (parasitized and uninfected)
- Real-time prediction with confidence levels
- Downloadable prediction summaries

## Model Development

For a detailed walkthrough of the machine learning workflow that led to this model, see the Jupyter notebook:

**📓 [03_final_resnet50.ipynb](notebooks/03_final_resnet50.ipynb)**

This notebook demonstrates the complete experimentation process:
- First attempt: Baseline CNN
- Tuning attempt: Hyperparameter optimization
- Transfer learning attempt: ResNet50 fine-tuning
- Final training run: Production model development
- Model performance metrics in sidebar

## Model Details

- **Architecture**: ResNet50 (pre-trained on ImageNet, fine-tuned for binary classification)
- **Input Size**: 128 x 128 RGB images
- **Task**: Binary classification (Parasitized vs Uninfected)
- **Output**: Probability of uninfected (sigmoid activation)
- **Validation Accuracy**: 97.88%
- **Validation Loss**: 0.0588

## Model Development Journey

The model development process involved iterative experimentation and improvement:

| Model | Validation Accuracy | Notes |
|-------|-------------------|--------|
| Baseline CNN | 53% | Underfit, weaker generalization |
| Improved CNN | 92.8% | Stronger but less robust |
| Fine-tuned ResNet50 | 97.88% | Best final model selected |

This project demonstrates a systematic, evidence-based approach to machine learning development, showcasing the iterative process of experimentation, evaluation, and improvement that leads to production-ready models.

### 📊 Development Philosophy
- **Systematic experimentation**: Each model iteration builds on lessons learned from previous attempts
- **Evidence-based decisions**: Model selection based on quantitative metrics and qualitative analysis
- **Portfolio structure**: Clear separation between research artifacts and production deployment
- **Reproducibility**: Documented notebooks showing the complete development workflow

### 📁 Project Structure
```
malaria-detection-ai/
├── models/                    # Production and development models
│   ├── best_malaria_model.keras    # Production model (ResNet50)
│   ├── baseline_cnn.keras          # Development artifact
│   └── improved_cnn.keras          # Development artifact
├── notebooks/                 # Research and experimentation
│   ├── 01_baseline_model.ipynb     # Initial CNN development
│   ├── 02_model_tuning.ipynb       # Architecture improvements
│   └── 03_final_resnet50.ipynb     # Transfer learning approach
├── assets/                    # Charts and evaluation plots used by the app
├── visuals/                   # Optional legacy plot location still supported by the app
└── app.py                     # Production Streamlit application
```

### 🔬 Experimental Progression

#### Phase 1: Baseline CNN (`01_baseline_model.ipynb`)
- **Goal**: Establish a working baseline with minimal complexity
- **Architecture**: Simple 2-layer CNN (32→64 filters)
- **Result**: 53% validation accuracy
- **Lessons**: Underfitting, poor generalization, need for more capacity

#### Phase 2: Model Tuning (`02_model_tuning.ipynb`)
- **Goal**: Improve architecture and training through systematic experimentation
- **Improvements**: Deeper network (32→64→128→256), batch normalization, dropout, data augmentation
- **Hyperparameter tuning**: Learning rate optimization, regularization parameters
- **Result**: 92.8% validation accuracy (+39.8 percentage points)
- **Lessons**: Data augmentation and regularization dramatically improve generalization

#### Phase 3: Transfer Learning (`03_final_resnet50.ipynb`)
- **Goal**: Leverage pre-trained features for state-of-the-art performance
- **Approach**: Fine-tune ResNet50 pre-trained on ImageNet
- **Result**: 97.88% validation accuracy (+5 percentage points)
- **Lessons**: Transfer learning provides superior feature extraction

### 🎯 Model Selection Rationale
The fine-tuned ResNet50 was selected as the production model because:
- **Superior performance**: Highest validation accuracy (97.88%)
- **Better generalization**: Pre-trained features handle medical imaging nuances
- **Robustness**: Less sensitive to data variations compared to custom CNNs
- **Industry standard**: Transfer learning is the current best practice for image classification

### 📈 Key Insights Learned
1. **Start simple**: Baseline models provide crucial benchmarks for improvement
2. **Data augmentation is critical**: Transforms dramatically improve generalization
3. **Transfer learning outperforms custom architectures**: Pre-trained models capture complex features
4. **Systematic documentation matters**: Notebooks create reproducible, portfolio-worthy work
5. **Production ≠ Research**: Separate deployment models from experimental artifacts

This development journey demonstrates ML maturity through systematic experimentation, evidence-based decision making, and professional project organization.
- Achieved 97.88% validation accuracy with 0.0588 validation loss
- Superior performance due to transfer learning from ImageNet
- Better handling of medical imaging characteristics compared to custom CNNs
- Robust feature extraction for distinguishing parasitized vs uninfected cells

The ResNet50 model won due to its proven architecture for image classification tasks, transfer learning benefits, and superior validation metrics compared to the custom CNN approaches.

## Prediction Logic and Uncertainty Thresholds

The model outputs a probability value between 0 and 1, representing the likelihood that the cell is **uninfected**.

### Confidence Levels
- **High Confidence**: When the maximum probability (uninfected or parasitized) ≥ 0.85
- **Moderate Confidence**: When the maximum probability ≥ 0.65
- **Low Confidence / Manual Review Recommended**: When the maximum probability < 0.65

### Classification Thresholds
- **High-confidence Uninfected**: Raw prediction ≥ 0.75
- **High-confidence Parasitized**: Raw prediction ≤ 0.25
- **Uncertain Zone**: Raw prediction between 0.25 and 0.75

### Rationale for Thresholds
Instead of using a simple 0.5 threshold (which would classify uninfected if > 0.5, parasitized if < 0.5), this app uses asymmetric thresholds to prioritize safety in a medical context:

- **Conservative Approach**: Requires high confidence (≥75% for uninfected, ≤25% for parasitized) to make definitive predictions
- **Avoids False Negatives**: A cell with 60% uninfected probability won't be classified as uninfected, reducing risk of missing infected cells
- **Manual Review Zone**: Predictions between 25-75% require human review, ensuring uncertain cases aren't automated
- **Reduces Overconfident Borderline Predictions**: The uncertain zone prevents the model from making definitive calls on borderline cases where it might be overconfident despite being close to 50/50
- **Medical Safety**: In healthcare applications, it's better to flag uncertain cases for expert review than risk incorrect automated diagnoses

This design choice reflects the high stakes of medical AI, where false confidence can be more dangerous than admitting uncertainty.

### Interpretation
- Raw prediction close to 1.0 → High confidence uninfected
- Raw prediction close to 0.0 → High confidence parasitized
- Raw prediction around 0.5 → Uncertain, manual review recommended

## Usage

1. Create and activate a virtual environment:
   `python -m venv .venv`
   `.\.venv\Scripts\Activate.ps1`
2. Install dependencies:
   `python -m pip install --upgrade pip`
   `pip install -r requirements.txt`
3. Run the app:
   `streamlit run app.py`
4. Upload an image or use sample buttons
5. View the prediction report and model evaluation visuals

Recommended Python version: 3.10 or 3.11 for the best TensorFlow compatibility on Windows.

## Streamlit Community Cloud Deployment

Use these deployment settings in Streamlit Community Cloud:

1. Repository: `jesspala77/malaria-detection-ai`
2. Branch: `main`
3. Entrypoint file: `model/app.py`
4. Python version: `3.10` or `3.11`

Important:

- Community Cloud can fail on this project if it deploys with a newer Python version such as `3.14`.
- The failure happens during dependency installation because `tensorflow>=2.16,<2.19` does not have compatible wheels for Python `3.14`.
- If you already created the app with the wrong Python version, delete it and redeploy it with Python `3.10` or `3.11`.

## Disclaimer

This application is for research and educational purposes only. It should not be used for clinical diagnosis. Always consult medical professionals for actual malaria screening.

## Built With

- Streamlit
- TensorFlow/Keras
- ResNet50
- PIL (Pillow)

## Author

Jessica Palacio
