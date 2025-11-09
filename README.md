# 🏥 ECG AI Model System

A comprehensive, modular ECG classification system built with PyTorch. This system provides separate, organized components for training, prediction, and model management.

## 🎯 Features

- **🧠 Deep Learning Models**: CNN and LSTM architectures for ECG classification
- **📊 Multi-label Classification**: Predict multiple diagnostic classes simultaneously
- **💾 Model Persistence**: Save and load trained models with PyTorch
- **🔮 Prediction Interface**: Interactive prediction on new ECG data
- **📈 Performance Metrics**: Comprehensive evaluation with F1-scores
- **🛠️ Modular Design**: Separate files for different functionalities
- **🚀 Easy Startup**: Single main script to access all features

## 📁 Project Structure

```
ECG AI MODEL EGC/
├── main.py                    # 🚀 Main startup script
├── requirements.txt           # 📦 Dependencies
├── README.md                 # 📖 Documentation
├── models/                   # 🧠 Model architectures
│   └── ecg_cnn.py           # CNN and LSTM models
├── data/                     # 📊 Data handling
│   └── ecg_dataset.py       # Dataset and data loaders
├── training/                 # 🚀 Training scripts
│   └── train_ecg_model.py   # Model training
├── prediction/              # 🔮 Prediction scripts
│   └── predict_ecg.py       # Model prediction
├── utils/                    # 🛠️ Utilities
│   └── model_utils.py       # Model saving/loading
├── saved_models/             # 💾 Trained models
└── DATA/                     # 📁 ECG data directory
    ├── ptbxl_database.csv
    ├── scp_statements.csv
    ├── records100/
    └── records500/
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

Ensure your ECG data is in the `./DATA/` directory with:
- `ptbxl_database.csv` - ECG annotations
- `scp_statements.csv` - Diagnostic statements
- `records100/` - 100Hz ECG recordings
- `records500/` - 500Hz ECG recordings

### 3. Run the System

```bash
# Start the main interface
python main.py
```

## 🎮 Usage

### Main Menu Options

1. **🚀 Train a new ECG model** - Train CNN or LSTM models
2. **🔮 Make predictions on new data** - Interactive prediction interface
3. **📊 Analyze existing model performance** - Model evaluation
4. **🛠️ Model management** - List, delete, or manage saved models
5. **📋 View system information** - System status and requirements
6. **❓ Help and documentation** - Usage guide
7. **🚪 Exit** - Exit the system

### Training a Model

```bash
# Option 1: Use main interface
python main.py
# Then select option 1

# Option 2: Direct training
python training/train_ecg_model.py
```

### Making Predictions

```bash
# Option 1: Use main interface
python main.py
# Then select option 2

# Option 2: Direct prediction
python prediction/predict_ecg.py
```

## 🧠 Model Architectures

### CNN Model (`ECGCNN`)
- **Input**: 12-lead ECG signals (1000 samples at 100Hz)
- **Architecture**: 4 convolutional layers + 3 fully connected layers
- **Features**: Batch normalization, dropout, max pooling
- **Output**: Multi-label classification with sigmoid activation

### LSTM Model (`ECGLSTM`)
- **Input**: 12-lead ECG signals
- **Architecture**: Bidirectional LSTM + fully connected layers
- **Features**: Dropout regularization
- **Output**: Multi-label classification

## 📊 Supported Diagnostic Classes

The system can classify ECG signals into these diagnostic superclasses:

- **NORM**: Normal ECG
- **MI**: Myocardial Infarction
- **STTC**: ST/T changes
- **CD**: Conduction Disturbance
- **HYP**: Hypertrophy

## 🔧 Configuration

### Training Parameters

```python
config = {
    'model_type': 'cnn',        # 'cnn' or 'lstm'
    'epochs': 30,               # Training epochs
    'learning_rate': 0.001,     # Learning rate
    'batch_size': 16,           # Batch size
    'patience': 10,             # Early stopping patience
    'data_path': './DATA/',     # Data directory
    'sampling_rate': 100,       # ECG sampling rate
    'save_dir': './saved_models/' # Model save directory
}
```

### Model Saving

Models are automatically saved with:
- Model state dictionary
- Training metadata
- Class names
- Performance metrics
- Timestamp

## 📈 Performance Metrics

The system provides comprehensive evaluation:

- **F1-Score** (Macro and Micro)
- **Precision and Recall**
- **Per-class metrics**
- **Confusion matrices**
- **ROC curves**

## 🛠️ Advanced Usage

### Custom Model Training

```python
from training.train_ecg_model import ECGTrainer

# Initialize trainer
trainer = ECGTrainer(model_type='cnn', save_dir='./my_models/')

# Train with custom parameters
results = trainer.train(
    epochs=50,
    learning_rate=0.0005,
    batch_size=32,
    patience=15
)
```

### Batch Prediction

```python
from prediction.predict_ecg import ECGPredictor

# Load model
predictor = ECGPredictor('./saved_models/best_model.pth')

# Predict on multiple samples
results = predictor.predict_batch(ecg_signals, threshold=0.5)
```

### Model Management

```python
from utils.model_utils import ModelSaver

# List all saved models
saver = ModelSaver('./saved_models/')
models = saver.list_models()

for model in models:
    print(f"Model: {model['filename']}")
    print(f"Epoch: {model['epoch']}")
    print(f"Val Loss: {model['val_loss']}")
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Error**: Reduce batch size or use smaller model
2. **mps Error**: Ensure PyTorch mps installation matches your GPU
3. **Data Loading Error**: Check DATA directory structure and file permissions
4. **Model Loading Error**: Ensure model file is not corrupted

### Performance Tips

1. **GPU Usage**: The system automatically uses GPU if available
2. **Batch Size**: Adjust based on GPU memory (16-64 recommended)
3. **Learning Rate**: Start with 0.001 and adjust based on convergence
4. **Early Stopping**: Use patience of 10-20 epochs to prevent overfitting

## 📚 API Reference

### ECGTrainer Class

```python
trainer = ECGTrainer(model_type='cnn', device=None, save_dir='./saved_models/')
results = trainer.train(epochs=30, learning_rate=0.001, batch_size=16, patience=10)
```

### ECGPredictor Class

```python
predictor = ECGPredictor(model_path='./saved_models/best_model.pth')
result = predictor.predict_single(ecg_signal, threshold=0.5)
```

### ECGDataLoader Class

```python
data_loader = ECGDataLoader(data_path='./DATA/', sampling_rate=100)
X, y, Y = data_loader.preprocess_data()
train_loader, val_loader, test_loader = data_loader.create_data_loaders(X, y)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source. Please ensure you have proper licenses for the ECG data you use.

## 📖 Citation

If you use this system in your research, please cite the PTB-XL dataset:

```
@article{ptb-xl2020,
  title={PTB-XL, a large publicly available electrocardiography dataset},
  author={Wagner, Patrick and Strodthoff, Nils and Bousseljot, R{\"u}diger and Kreiseler, Dieter and Lunze, Fatima I and Samek, Wojciech and Schaeffter, Tobias},
  journal={Scientific Data},
  year={2020}
}
```

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the documentation
3. Open an issue on GitHub
4. Contact the development team

---

**🏥 ECG AI Model System - Complete ECG Classification Solution**
# ECG-DeepNet
