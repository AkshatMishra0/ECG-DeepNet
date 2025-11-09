#!/usr/bin/env python3
"""
ECG AI Model System - Simple Demo
Demonstrates the key features of the system
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

def demo_system_overview():
    """Demonstrate system overview"""
    print("🏥 ECG AI MODEL SYSTEM - DEMO")
    print("=" * 50)
    print()
    print("📁 PROJECT STRUCTURE:")
    print("├── main.py                    # 🚀 Main startup script")
    print("├── models/ecg_cnn.py          # 🧠 CNN & LSTM architectures")
    print("├── data/ecg_dataset.py        # 📊 Data loading utilities")
    print("├── training/train_ecg_model.py # 🚀 Model training")
    print("├── prediction/predict_ecg.py  # 🔮 Model prediction")
    print("├── utils/model_utils.py       # 🛠️ Model utilities")
    print("├── saved_models/              # 💾 Trained models")
    print("└── DATA/                      # 📁 ECG data directory")
    print()

def demo_model_creation():
    """Demonstrate model creation"""
    print("🧠 MODEL CREATION DEMO:")
    print("-" * 30)
    
    try:
        # Import the model
        sys.path.insert(0, '.')
        from models.ecg_cnn import ECGCNN, ECGLSTM
        
        # Create CNN model
        cnn_model = ECGCNN(input_length=1000, num_classes=5)
        print("✅ CNN Model created successfully")
        
        # Create LSTM model  
        lstm_model = ECGLSTM(input_size=12, num_classes=5)
        print("✅ LSTM Model created successfully")
        
        # Get model info
        cnn_info = cnn_model.get_model_info()
        print(f"   📊 CNN Parameters: {cnn_info['total_parameters']:,}")
        print(f"   📊 Input Length: {cnn_info['input_length']}")
        print(f"   📊 Classes: {cnn_info['num_classes']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def demo_data_check():
    """Demonstrate data directory check"""
    print("\n📊 DATA DIRECTORY CHECK:")
    print("-" * 30)
    
    data_path = Path('./DATA')
    if data_path.exists():
        print("✅ DATA directory found")
        
        # Check for CSV files
        csv_files = list(data_path.glob('*.csv'))
        print(f"   📄 CSV files: {len(csv_files)}")
        
        # Check for required files
        required_files = ['ptbxl_database.csv', 'scp_statements.csv']
        for file in required_files:
            if (data_path / file).exists():
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file} (missing)")
        
        # Check for ECG records
        records_100 = data_path / 'records100'
        records_500 = data_path / 'records500'
        
        if records_100.exists():
            dat_files = list(records_100.rglob('*.dat'))
            print(f"   📊 ECG records (100Hz): {len(dat_files)} files")
        
        if records_500.exists():
            dat_files = list(records_500.rglob('*.dat'))
            print(f"   📊 ECG records (500Hz): {len(dat_files)} files")
        
        return True
    else:
        print("❌ DATA directory not found")
        return False

def demo_utilities():
    """Demonstrate utility functions"""
    print("\n🛠️ UTILITIES DEMO:")
    print("-" * 30)
    
    try:
        sys.path.insert(0, '.')
        from utils.model_utils import ModelSaver, TrainingMetrics
        
        # Test model saver
        saver = ModelSaver('./demo_models/')
        print("✅ Model saver created")
        
        # Test training metrics
        metrics = TrainingMetrics()
        metrics.update(1, 0.5, 0.6, 0.001)
        metrics.update(2, 0.4, 0.5, 0.0008)
        print("✅ Training metrics working")
        
        # Test model listing
        models = saver.list_models()
        print(f"   📊 Saved models: {len(models)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utilities failed: {e}")
        return False

def demo_system_info():
    """Demonstrate system information"""
    print("\n💻 SYSTEM INFORMATION:")
    print("-" * 30)
    
    try:
        print(f"🐍 Python: {sys.version.split()[0]}")
        print(f"🔥 PyTorch: {torch.__version__}")
        print(f"📊 NumPy: {np.__version__}")
        print(f"📈 Pandas: {torch.__version__}")
        print(f"🤖 mps: {'✅ Available' if torch.mps.is_available() else '❌ Not available'}")
        
        if torch.mps.is_available():
            print(f"   🎮 GPU: {torch.mps.get_device_name(0)}")
            print(f"   🎮 Memory: {torch.mps.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ System info failed: {e}")
        return False

def demo_usage_instructions():
    """Demonstrate usage instructions"""
    print("\n🚀 USAGE INSTRUCTIONS:")
    print("-" * 30)
    print("1. 🚀 Start the system:")
    print("   python main.py")
    print()
    print("2. 🧠 Train a model:")
    print("   python training/train_ecg_model.py")
    print()
    print("3. 🔮 Make predictions:")
    print("   python prediction/predict_ecg.py")
    print()
    print("4. 🧪 Run tests:")
    print("   python test_system.py")
    print()

def main():
    """Main demo function"""
    demo_system_overview()
    
    # Run demos
    demos = [
        ("Model Creation", demo_model_creation),
        ("Data Check", demo_data_check),
        ("Utilities", demo_utilities),
        ("System Info", demo_system_info)
    ]
    
    passed = 0
    for demo_name, demo_func in demos:
        if demo_func():
            passed += 1
    
    print(f"\n📊 DEMO RESULTS: {passed}/{len(demos)} components working")
    
    if passed == len(demos):
        print("🎉 SYSTEM IS READY TO USE!")
    else:
        print("⚠️  Some components need attention")
    
    demo_usage_instructions()
    
    return passed == len(demos)

if __name__ == "__main__":
    success = main()
    print(f"\n{'='*50}")
    if success:
        print("✅ ECG AI Model System is ready!")
        print("Run 'python main.py' to start using the system.")
    else:
        print("⚠️  Please check the issues above before using the system.")
