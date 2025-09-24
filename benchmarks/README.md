# MIMIC-III Benchmarks

This directory contains standardized benchmark implementations for clinical ML tasks using the MIMIC-III dataset. The benchmarks are based on established methodologies to ensure reproducible results and fair comparisons.

## Overview

The benchmark suite includes three primary clinical prediction tasks:
- **Mortality Prediction**: ICU mortality prediction using physiological time series
- **Length of Stay Prediction**: Hospital stay duration estimation
- **Readmission Risk Assessment**: 30-day readmission probability

## Repository Structure

```
benchmarks/
├── README.md                    # This file
├── mortality_prediction/        # ICU mortality prediction models
│   ├── models/                  # Model implementations
│   ├── preprocessing/           # Data preprocessing scripts
│   └── evaluation/              # Evaluation metrics and scripts
├── length_of_stay/             # LOS prediction models
│   ├── models/                 # Model implementations
│   ├── preprocessing/          # Data preprocessing scripts
│   └── evaluation/             # Evaluation metrics and scripts
└── readmission/                # Readmission risk models
    ├── models/                 # Model implementations
    ├── preprocessing/          # Data preprocessing scripts
    └── evaluation/             # Evaluation metrics and scripts
```

## Getting Started

### Prerequisites
1. **MIMIC-III Access**: Ensure you have completed the data setup following `../data/GETTING_STARTED.txt`
2. **Python Environment**: Python 3.8+ with required packages
3. **Dependencies**: Install requirements using `pip install -r ../requirements.txt`

### Quick Start

1. **Setup Data Paths**:
   ```bash
   export MIMIC_DATA_DIR=/path/to/mimic-iii/
   ```

2. **Run Preprocessing**:
   ```bash
   python mortality_prediction/preprocessing/preprocess.py
   python length_of_stay/preprocessing/preprocess.py
   python readmission/preprocessing/preprocess.py
   ```

3. **Train Models**:
   ```bash
   python mortality_prediction/models/train.py
   python length_of_stay/models/train.py
   python readmission/models/train.py
   ```

4. **Evaluate Results**:
   ```bash
   python mortality_prediction/evaluation/evaluate.py
   python length_of_stay/evaluation/evaluate.py
   python readmission/evaluation/evaluate.py
   ```

## Benchmark Tasks

### 1. In-Hospital Mortality Prediction

**Objective**: Predict whether a patient will die during their ICU stay

**Data Features**:
- Physiological time series (vital signs, lab values)
- First 48 hours of ICU stay
- Demographics and admission details

**Models Implemented**:
- Logistic Regression (baseline)
- Random Forest
- LSTM Networks
- Transformer-based models

**Evaluation Metrics**:
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)
- Accuracy, Precision, Recall, F1-score

**Baseline Performance**:
- Logistic Regression: AUROC ~0.85
- LSTM: AUROC ~0.87
- Transformer: AUROC ~0.89

### 2. Length of Stay Prediction

**Objective**: Predict the length of hospital stay for ICU patients

**Data Features**:
- Admission details and demographics
- Clinical measurements from first 24 hours
- Diagnosis and procedure codes

**Models Implemented**:
- Linear Regression (baseline)
- Random Forest Regressor
- LSTM Networks
- Attention-based models

**Evaluation Metrics**:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

**Baseline Performance**:
- Linear Regression: MAE ~3.2 days
- Random Forest: MAE ~2.8 days
- LSTM: MAE ~2.5 days

### 3. Readmission Risk Assessment

**Objective**: Predict 30-day hospital readmission probability

**Data Features**:
- Complete hospital stay information
- Discharge summaries and notes
- Historical admission patterns
- Medication prescriptions

**Models Implemented**:
- Logistic Regression (baseline)
- Gradient Boosting (XGBoost)
- LSTM with attention
- Clinical BERT fine-tuned model

**Evaluation Metrics**:
- AUROC and AUPRC
- Calibration metrics
- Clinical utility measures

**Baseline Performance**:
- Logistic Regression: AUROC ~0.68
- XGBoost: AUROC ~0.72
- Clinical BERT: AUROC ~0.75

## Reference Implementation

This benchmark suite is based on and compatible with:

- **YerevaNN MIMIC-III Benchmarks**: [https://github.com/YerevaNN/mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks)
- **Original Paper**: Harutyunyan, H., et al. "Multitask learning and benchmarking with clinical time series data." Scientific data 6.1 (2019): 1-18.

### Key Features
- Standardized data preprocessing pipelines
- Consistent evaluation protocols
- Reproducible experiment configurations
- Fair comparison baselines

## Data Preprocessing

### Common Preprocessing Steps
1. **Data Extraction**: Extract relevant tables from MIMIC-III database
2. **Cohort Selection**: Apply inclusion/exclusion criteria
3. **Feature Engineering**: Create time-series features and clinical indicators
4. **Normalization**: Standardize numerical features
5. **Missing Value Handling**: Imputation strategies for missing data
6. **Train/Validation/Test Split**: 70/15/15 split with temporal ordering

### Cohort Definitions

**Inclusion Criteria**:
- Adult patients (age ≥ 18)
- ICU stay duration ≥ 48 hours
- Sufficient clinical measurements available

**Exclusion Criteria**:
- Multiple ICU stays (use first stay only)
- Patients with missing critical demographics
- ICU stays with <80% data availability

## Model Training

### Training Configuration
- **Cross-validation**: 5-fold stratified CV for hyperparameter tuning
- **Early Stopping**: Monitor validation loss with patience=10
- **Regularization**: L1/L2 regularization, dropout for neural networks
- **Optimization**: Adam optimizer with learning rate scheduling

### Hyperparameter Search
- Grid search for traditional ML models
- Bayesian optimization for deep learning models
- 50 trials maximum per model type

## Evaluation Protocol

### Performance Metrics
1. **Classification Tasks** (Mortality, Readmission):
   - Primary: AUROC, AUPRC
   - Secondary: Accuracy, Precision, Recall, F1-score
   - Clinical: Sensitivity at fixed specificity levels

2. **Regression Tasks** (Length of Stay):
   - Primary: MAE, RMSE
   - Secondary: R², MAPE
   - Clinical: Prediction within ±1 day accuracy

### Statistical Testing
- Bootstrap confidence intervals (n=1000)
- McNemar's test for model comparison
- Wilcoxon signed-rank test for regression metrics

### Clinical Validation
- Subgroup analysis by age, gender, admission type
- Temporal validation on recent data
- External validation on other ICU datasets (when available)

## Usage Examples

### Training a Mortality Prediction Model

```python
from mortality_prediction.models import LSTMModel
from mortality_prediction.preprocessing import MIMICPreprocessor

# Load and preprocess data
preprocessor = MIMICPreprocessor(data_path='/path/to/mimic')
X_train, y_train, X_val, y_val = preprocessor.load_mortality_data()

# Initialize and train model
model = LSTMModel(input_dim=76, hidden_dim=128, num_layers=2)
model.fit(X_train, y_train, validation_data=(X_val, y_val))

# Evaluate model
metrics = model.evaluate(X_test, y_test)
print(f"AUROC: {metrics['auroc']:.3f}")
```

### Custom Model Integration

```python
from benchmarks.base import BaseModel, BenchmarkEvaluator

class CustomModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your model
    
    def fit(self, X, y, **kwargs):
        # Training logic
        pass
    
    def predict_proba(self, X):
        # Prediction logic
        return predictions

# Evaluate using standard protocol
evaluator = BenchmarkEvaluator(task='mortality')
results = evaluator.evaluate_model(CustomModel(), X_test, y_test)
```

## Contributing

### Adding New Models
1. Follow the established directory structure
2. Inherit from `BaseModel` class for consistency
3. Include comprehensive documentation
4. Add unit tests for model components
5. Update benchmark results in this README

### Reporting Results
When reporting benchmark results, please include:
- Model architecture and hyperparameters
- Training time and computational requirements
- Confidence intervals for all metrics
- Code for reproducibility

## Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce batch size or use data generators
2. **Slow Training**: Enable GPU acceleration if available
3. **Poor Performance**: Check data preprocessing and feature engineering
4. **Reproducibility Issues**: Set random seeds and ensure deterministic operations

### Performance Optimization
- Use mixed precision training for faster convergence
- Implement efficient data loading with prefetching
- Consider model ensembles for improved performance
- Profile code to identify bottlenecks

## Citation

If you use these benchmarks in your research, please cite:

```bibtex
@article{harutyunyan2019multitask,
  title={Multitask learning and benchmarking with clinical time series data},
  author={Harutyunyan, Hrayr and Khachatrian, Hrant and Kale, David C and Ver Steeg, Greg and Galstyan, Aram},
  journal={Scientific data},
  volume={6},
  number={1},
  pages={1--18},
  year={2019},
  publisher={Nature Publishing Group}
}
```

## References

- [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/)
- [YerevaNN MIMIC-III Benchmarks](https://github.com/YerevaNN/mimic3-benchmarks)
- [Clinical ML Benchmarking Best Practices](https://www.nature.com/articles/s41597-019-0103-9)
- [Time Series Classification in Healthcare](https://arxiv.org/abs/1909.07782)

## Support

For questions and support:
- Check existing GitHub issues
- Review the troubleshooting section
- Consult MIMIC-III documentation
- Contact the maintainers for specific implementation questions

---

**Note**: This benchmark suite is designed for research purposes. Ensure proper validation and testing before using any models in clinical settings.
