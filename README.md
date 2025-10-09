# 🏥 Drug Classification ML Pipeline

A comprehensive Machine Learning pipeline for drug recommendation with continuous integration and deployment to Hugging Face Spaces.

## 🎯 Project Overview

This project demonstrates a complete ML pipeline for drug classification based on patient characteristics:
- **Age**: Patient's age (15-74 years)
- **Gender**: Patient's gender (M/F)
- **Blood Pressure**: Current blood pressure level (HIGH/LOW/NORMAL)
- **Cholesterol**: Current cholesterol level (HIGH/NORMAL)
- **Na_to_K Ratio**: Sodium to Potassium ratio in blood (6.2-38.2)

The model predicts the most suitable drug from 5 categories: drugA, drugB, drugC, drugX, DrugY.

## 📁 Project Structure

```
├── data/                    # Dataset storage
│   └── drug200.csv         # Drug classification dataset
├── Model/                   # Saved trained models
│   └── drug_pipeline.joblib # Trained RandomForest pipeline
├── app/                     # Gradio web application
│   ├── drug_app.py         # Main app file
│   ├── README.md           # HuggingFace metadata
│   └── requirements.txt    # App dependencies
├── Results/                 # Model evaluation outputs
│   ├── metrics.txt         # Performance metrics
│   └── model_results.png   # Confusion matrix visualization
├── .github/workflows/       # CI/CD configuration
│   └── ci.yml              # GitHub Actions workflow
├── train.py                # Model training script
├── notebook.ipynb          # Experimentation notebook
├── Makefile                # Automation commands
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/mritula2311/ML-MODEL-DRUG-RECOMENDATION.git
cd ML-MODEL-DRUG-RECOMENDATION
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Model
```bash
python train.py
```

### 4. Run Gradio App
```bash
cd app
python drug_app.py
```

## 🔄 CI/CD Pipeline

The project includes automated CI/CD using GitHub Actions:

### Workflow Steps:
1. **Code Push** → GitHub repository
2. **Auto Trigger** → GitHub Actions workflow
3. **Environment Setup** → Ubuntu + Python 3.9
4. **Install Dependencies** → `make install`
5. **Train Model** → `make train`
6. **Evaluate Model** → `make eval`
7. **Upload Artifacts** → Model + Results
8. **Update Branch** → Create update branch
9. **Deploy to HF** → Hugging Face Spaces
10. **Create Release** → Version tagged release

### Makefile Commands:
```bash
make install        # Install dependencies
make train          # Train the model
make eval           # Evaluate and create report
make deploy         # Deploy to Hugging Face
make run-app        # Run Gradio app locally
make pipeline       # Full pipeline (install + train + eval)
```

## 🤗 Hugging Face Deployment

- **Space URL**: https://huggingface.co/spaces/Mritula123/Mlmodeldrug
- **SDK**: Gradio 4.16.0
- **Auto-deployed** via GitHub Actions on push to main

### Setup Secrets:
1. `HF_TOKEN`: Hugging Face API token with write permissions
2. `GITHUB_TOKEN`: Automatically provided by GitHub Actions

## 🧪 Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 5 input features with preprocessing pipeline
- **Preprocessing**: 
  - Categorical encoding (OrdinalEncoder)
  - Missing value imputation (SimpleImputer)
  - Numerical scaling (StandardScaler)
- **Performance**: High accuracy with comprehensive evaluation metrics

## 📊 Results

Model performance metrics and confusion matrix are automatically generated and saved to the `Results/` folder:
- `metrics.txt`: Accuracy and F1 scores
- `model_results.png`: Confusion matrix visualization

## 🛠️ Development

### Local Development:
1. Edit code in your preferred IDE
2. Test locally using `make pipeline`
3. Push to GitHub to trigger CI/CD

### Jupyter Notebook:
Use `notebook.ipynb` for experimentation and development:
- Data exploration
- Model experimentation
- Pipeline testing
- Visualization

## 📝 Requirements

### Core Dependencies:
- scikit-learn: ML algorithms and preprocessing
- pandas: Data manipulation
- numpy: Numerical computing
- matplotlib: Visualization
- joblib: Model serialization
- gradio: Web interface
- huggingface_hub: HF integration

### Development Dependencies:
- cml: ML experiment tracking
- GitHub Actions: CI/CD automation

## 🔒 Security

- Sensitive tokens stored as GitHub Secrets
- No hardcoded credentials in code
- Automated security scanning via GitHub Actions

## 📄 License

Apache License 2.0

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test locally
4. Submit a pull request
5. CI/CD will automatically test and deploy

## ⚠️ Disclaimer

This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

---

**Built  using Scikit-learn, Gradio, and automated CI/CD**
