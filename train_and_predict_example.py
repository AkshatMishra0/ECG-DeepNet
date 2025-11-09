#!/usr/bin/env python3
"""
ECG AI Model Training and Prediction Example

This script demonstrates how to:
1. Train an AI model on ECG data
2. Make predictions on new ECG data
3. Save and load trained models
"""

import numpy as np
import pandas as pd
from ecg_ai_trainer import ECGAITrainer
import matplotlib.pyplot as plt
import seaborn as sns

def train_new_model():
    """Train a new ECG classification model"""
    print("🚀 Starting ECG AI Model Training")
    print("=" * 50)
    
    # Initialize the trainer
    trainer = ECGAITrainer(
        data_path='./DATA/',
        sampling_rate=100
    )
    
    # Train the model with custom parameters
    print("📊 Training model with the following parameters:")
    print("   - Epochs: 30")
    print("   - Learning Rate: 0.001")
    print("   - Batch Size: 16")
    print("   - Early Stopping: 10 epochs patience")
    print()
    
    model = trainer.train_model(
        epochs=30,
        learning_rate=0.001,
        batch_size=16,
        patience=10
    )
    
    # Save the trained model
    trainer.save_model('trained_ecg_model.pkl')
    print("✅ Model training completed and saved!")
    
    return trainer

def load_existing_model():
    """Load a previously trained model"""
    print("📁 Loading existing trained model...")
    
    trainer = ECGAITrainer(data_path='./DATA/')
    try:
        trainer.load_model('trained_ecg_model.pkl')
        print("✅ Model loaded successfully!")
        return trainer
    except FileNotFoundError:
        print("❌ No trained model found. Please train a model first.")
        return None

def predict_on_new_data(trainer, num_samples=5):
    """Make predictions on new ECG data"""
    print(f"\n🔮 Making predictions on {num_samples} new ECG samples...")
    
    # Load some sample data for demonstration
    Y = pd.read_csv('./DATA/ptbxl_database.csv', index_col='ecg_id')
    sample_indices = np.random.choice(len(Y), num_samples, replace=False)
    sample_data = trainer.load_raw_data(Y.iloc[sample_indices], 100, './DATA/')
    
    print("Sample predictions:")
    print("-" * 40)
    
    for i, (sample, idx) in enumerate(zip(sample_data, sample_indices)):
        # Make prediction
        predictions, probabilities = trainer.predict(sample)
        
        print(f"\nSample {i+1} (ECG ID: {Y.index[idx]}):")
        print(f"  Predicted classes: {predictions[0]}")
        print(f"  Confidence scores: {probabilities[0]}")
        
        # Show top 3 most confident predictions
        top_indices = np.argsort(probabilities[0])[-3:][::-1]
        print("  Top 3 predictions:")
        for j, idx in enumerate(top_indices):
            class_name = trainer.class_names[idx]
            confidence = probabilities[0][idx]
            print(f"    {j+1}. {class_name}: {confidence:.3f}")

def analyze_model_performance(trainer):
    """Analyze and visualize model performance"""
    print("\n📈 Analyzing model performance...")
    
    # Load test data
    Y = pd.read_csv('./DATA/ptbxl_database.csv', index_col='ecg_id')
    X = trainer.load_raw_data(Y, 100, './DATA/')
    
    # Get test fold data
    test_fold = 10
    test_mask = Y.strat_fold == test_fold
    X_test = X[test_mask]
    y_test = Y[test_mask]
    
    # Make predictions on test set
    all_predictions = []
    all_probabilities = []
    
    print("Making predictions on test set...")
    for i, sample in enumerate(X_test[:100]):  # Limit to 100 samples for demo
        predictions, probabilities = trainer.predict(sample)
        all_predictions.extend(predictions)
        all_probabilities.extend(probabilities)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/100 samples...")
    
    # Analyze class distribution
    print("\n📊 Class Distribution Analysis:")
    print("-" * 30)
    
    # Count predictions per class
    class_counts = {}
    for pred in all_predictions:
        for class_name in pred:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count} predictions")

def interactive_prediction():
    """Interactive prediction interface"""
    print("\n🎯 Interactive ECG Prediction")
    print("=" * 40)
    
    # Try to load existing model
    trainer = load_existing_model()
    if trainer is None:
        print("Training a new model first...")
        trainer = train_new_model()
    
    while True:
        print("\nOptions:")
        print("1. Predict on random samples")
        print("2. Analyze model performance")
        print("3. Show available classes")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            try:
                num_samples = int(input("How many samples to predict? (default 5): ") or "5")
                predict_on_new_data(trainer, num_samples)
            except ValueError:
                print("Invalid input. Using default of 5 samples.")
                predict_on_new_data(trainer, 5)
                
        elif choice == '2':
            analyze_model_performance(trainer)
            
        elif choice == '3':
            print(f"\nAvailable diagnostic classes ({len(trainer.class_names)}):")
            for i, class_name in enumerate(trainer.class_names, 1):
                print(f"  {i}. {class_name}")
                
        elif choice == '4':
            print("👋 Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

def main():
    """Main function to run the ECG AI trainer example"""
    print("🏥 ECG AI Model Trainer & Predictor")
    print("=" * 50)
    print("This tool can:")
    print("• Train AI models on ECG data")
    print("• Make predictions on new ECG data")
    print("• Analyze model performance")
    print("• Save and load trained models")
    print()
    
    # Check if model exists
    import os
    if os.path.exists('trained_ecg_model.pkl'):
        print("✅ Found existing trained model!")
        trainer = load_existing_model()
    else:
        print("🆕 No existing model found. Training a new one...")
        trainer = train_new_model()
    
    # Run interactive prediction
    interactive_prediction()

if __name__ == "__main__":
    main()
