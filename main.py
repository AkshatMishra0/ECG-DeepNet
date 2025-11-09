#!/usr/bin/env python3
"""
ECG AI Model System - Main Startup Script
Complete ECG classification system with separate modules
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print system banner"""
    print("🏥" + "=" * 48 + "🏥")
    print("    ECG AI MODEL SYSTEM - COMPLETE SOLUTION")
    print("🏥" + "=" * 48 + "🏥")
    print()

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        ('torch', 'torch'),
        ('numpy', 'numpy'), 
        ('pandas', 'pandas'),
        ('sklearn', 'scikit-learn'),
        ('wfdb', 'wfdb'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn')
    ]
    
    missing_packages = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nTo install missing packages, run:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_data_directory():
    """Check if data directory exists and has required files"""
    data_path = Path('./DATA')
    
    if not data_path.exists():
        print("❌ DATA directory not found!")
        print("Please ensure your ECG data is in the ./DATA/ directory")
        return False
    
    required_files = [
        'ptbxl_database.csv',
        'scp_statements.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not (data_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required data files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ Data directory and files found")
    return True

def show_menu():
    """Show main menu options"""
    print("\n🎯 MAIN MENU - Choose an option:")
    print("=" * 50)
    print("1. 🚀 Train a new ECG model")
    print("2. 🔮 Make predictions on new data")
    print("3. 📊 Analyze existing model performance")
    print("4. 🛠️  Model management (list, delete models)")
    print("5. 📋 View system information")
    print("6. ❓ Help and documentation")
    print("7. 🚪 Exit")
    print("=" * 50)

def train_model():
    """Start model training"""
    print("\n🚀 Starting ECG Model Training")
    print("-" * 40)
    
    # Check if training script exists
    training_script = Path('./training/train_ecg_model.py')
    if not training_script.exists():
        print("❌ Training script not found!")
        return
    
    try:
        # Run training script
        result = subprocess.run([sys.executable, str(training_script)], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Training completed successfully!")
        else:
            print("\n❌ Training failed!")
            
    except Exception as e:
        print(f"❌ Error running training: {e}")

def make_predictions():
    """Start prediction interface"""
    print("\n🔮 Starting ECG Prediction Interface")
    print("-" * 40)
    
    # Check if prediction script exists
    prediction_script = Path('./prediction/predict_ecg.py')
    if not prediction_script.exists():
        print("❌ Prediction script not found!")
        return
    
    try:
        # Run prediction script
        subprocess.run([sys.executable, str(prediction_script)], 
                      capture_output=False, text=True)
        
    except Exception as e:
        print(f"❌ Error running prediction: {e}")

def analyze_model():
    """Analyze model performance"""
    print("\n📊 Model Performance Analysis")
    print("-" * 40)
    
    # Check for saved models
    models_dir = Path('./saved_models')
    if not models_dir.exists() or not list(models_dir.glob('*.pth')):
        print("❌ No trained models found!")
        print("Please train a model first using option 1.")
        return
    
    print("Available models:")
    models = list(models_dir.glob('*.pth'))
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model.name}")
    
    try:
        choice = int(input(f"\nSelect model to analyze (1-{len(models)}): ")) - 1
        selected_model = models[choice]
        
        print(f"\nAnalyzing model: {selected_model.name}")
        print("This feature will be implemented in future versions.")
        print("For now, you can use the prediction interface to test your model.")
        
    except (ValueError, IndexError):
        print("Invalid selection.")

def model_management():
    """Model management interface"""
    print("\n🛠️  Model Management")
    print("-" * 40)
    
    models_dir = Path('./saved_models')
    if not models_dir.exists():
        print("No saved models directory found.")
        return
    
    models = list(models_dir.glob('*.pth'))
    if not models:
        print("No saved models found.")
        return
    
    print("Saved models:")
    for i, model in enumerate(models, 1):
        size_mb = model.stat().st_size / (1024 * 1024)
        print(f"  {i}. {model.name} ({size_mb:.1f} MB)")
    
    print("\nOptions:")
    print("1. Delete a model")
    print("2. View model details")
    print("3. Back to main menu")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == '1':
        try:
            model_choice = int(input(f"Select model to delete (1-{len(models)}): ")) - 1
            model_to_delete = models[model_choice]
            confirm = input(f"Delete {model_to_delete.name}? (y/n): ").strip().lower()
            
            if confirm == 'y':
                model_to_delete.unlink()
                print(f"✅ Deleted {model_to_delete.name}")
            else:
                print("Deletion cancelled.")
                
        except (ValueError, IndexError):
            print("Invalid selection.")
    
    elif choice == '2':
        print("Model details feature coming soon!")
    
    elif choice == '3':
        return
    
    else:
        print("Invalid choice.")

def show_system_info():
    """Show system information"""
    print("\n📋 System Information")
    print("-" * 40)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check PyTorch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"mps available: {torch.mps.is_available()}")
        if torch.mps.is_available():
            print(f"mps version: {torch.version.mps}")
            print(f"GPU count: {torch.mps.device_count()}")
    except ImportError:
        print("PyTorch: Not installed")
    
    # Check data directory
    data_path = Path('./DATA')
    if data_path.exists():
        csv_files = list(data_path.glob('*.csv'))
        print(f"Data files: {len(csv_files)} CSV files found")
        
        # Check for ECG records
        records_100 = data_path / 'records100'
        records_500 = data_path / 'records500'
        if records_100.exists():
            print(f"ECG records (100Hz): {len(list(records_100.rglob('*.dat')))} files")
        if records_500.exists():
            print(f"ECG records (500Hz): {len(list(records_500.rglob('*.dat')))} files")
    else:
        print("Data directory: Not found")
    
    # Check saved models
    models_dir = Path('./saved_models')
    if models_dir.exists():
        models = list(models_dir.glob('*.pth'))
        print(f"Saved models: {len(models)} models")
    else:
        print("Saved models: None")

def show_help():
    """Show help and documentation"""
    print("\n❓ Help and Documentation")
    print("-" * 40)
    print("This ECG AI Model System provides:")
    print()
    print("🧠 MODEL TRAINING:")
    print("   - Train CNN or LSTM models on ECG data")
    print("   - Automatic data preprocessing and validation")
    print("   - Early stopping and model saving")
    print()
    print("🔮 PREDICTIONS:")
    print("   - Make predictions on new ECG data")
    print("   - Support for single samples or batch processing")
    print("   - Interactive prediction interface")
    print()
    print("📊 FEATURES:")
    print("   - Multi-label classification")
    print("   - GPU acceleration support")
    print("   - Model persistence and loading")
    print("   - Comprehensive evaluation metrics")
    print()
    print("📁 FILE STRUCTURE:")
    print("   models/     - Model architectures")
    print("   data/       - Data loading utilities")
    print("   training/   - Training scripts")
    print("   prediction/ - Prediction scripts")
    print("   utils/      - Utility functions")
    print("   saved_models/ - Trained models")
    print()
    print("🚀 QUICK START:")
    print("   1. Ensure your ECG data is in ./DATA/ directory")
    print("   2. Choose option 1 to train a model")
    print("   3. Choose option 2 to make predictions")
    print()
    print("📖 For detailed documentation, see README.md")

def main():
    """Main function"""
    print_banner()
    
    # Check system requirements
    if not check_requirements():
        print("\nPlease install missing packages and try again.")
        return
    
    if not check_data_directory():
        print("\nPlease ensure your data is properly set up and try again.")
        return
    
    print("✅ System ready!")
    
    # Main loop
    while True:
        show_menu()
        
        try:
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == '1':
                train_model()
            elif choice == '2':
                make_predictions()
            elif choice == '3':
                analyze_model()
            elif choice == '4':
                model_management()
            elif choice == '5':
                show_system_info()
            elif choice == '6':
                show_help()
            elif choice == '7':
                print("\n👋 Thank you for using ECG AI Model System!")
                print("Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-7.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
