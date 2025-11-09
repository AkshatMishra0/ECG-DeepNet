#!/usr/bin/env python3
"""
Test script to demonstrate the ECG AI Model System functionality
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test if all modules can be imported"""
    print("🧪 Testing module imports...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, '.')
        
        # Test model imports
        from models.ecg_cnn import ECGCNN, ECGLSTM
        print("✅ Model architectures imported successfully")
        
        # Test data imports
        from data.ecg_dataset import ECGDataLoader, ECGDataset
        print("✅ Data loading modules imported successfully")
        
        # Test training imports
        from training.train_ecg_model import ECGTrainer
        print("✅ Training modules imported successfully")
        
        # Test prediction imports
        from prediction.predict_ecg import ECGPredictor
        print("✅ Prediction modules imported successfully")
        
        # Test utility imports
        from utils.model_utils import ModelSaver, TrainingMetrics, ModelEvaluator
        print("✅ Utility modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("\n🧪 Testing model creation...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, '.')
        
        from models.ecg_cnn import ECGCNN, ECGLSTM
        
        # Test CNN model
        cnn_model = ECGCNN(input_length=1000, num_classes=5)
        print("✅ CNN model created successfully")
        
        # Test LSTM model
        lstm_model = ECGLSTM(input_size=12, num_classes=5)
        print("✅ LSTM model created successfully")
        
        # Test model info
        cnn_info = cnn_model.get_model_info()
        print(f"   CNN parameters: {cnn_info['total_parameters']:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_data_loader():
    """Test data loader functionality"""
    print("\n🧪 Testing data loader...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, '.')
        
        from data.ecg_dataset import ECGDataLoader
        
        # Initialize data loader
        data_loader = ECGDataLoader(data_path='./DATA/', sampling_rate=100)
        print("✅ Data loader initialized successfully")
        
        # Test if data files exist
        data_path = Path('./DATA')
        if data_path.exists():
            csv_files = list(data_path.glob('*.csv'))
            print(f"✅ Found {len(csv_files)} CSV files in DATA directory")
            
            # Check for required files
            required_files = ['ptbxl_database.csv', 'scp_statements.csv']
            missing_files = [f for f in required_files if not (data_path / f).exists()]
            
            if missing_files:
                print(f"⚠️  Missing files: {missing_files}")
                return False
            else:
                print("✅ All required data files found")
                return True
        else:
            print("❌ DATA directory not found")
            return False
            
    except Exception as e:
        print(f"❌ Data loader error: {e}")
        return False

def test_utilities():
    """Test utility functions"""
    print("\n🧪 Testing utilities...")
    
    try:
        from utils.model_utils import ModelSaver, TrainingMetrics
        
        # Test model saver
        saver = ModelSaver('./test_models/')
        print("✅ Model saver created successfully")
        
        # Test training metrics
        metrics = TrainingMetrics()
        metrics.update(1, 0.5, 0.6, 0.001)
        print("✅ Training metrics working")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility error: {e}")
        return False

def test_system_info():
    """Test system information"""
    print("\n🧪 Testing system information...")
    
    try:
        import torch
        import numpy as np
        import pandas as pd
        import sklearn
        
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ NumPy version: {np.__version__}")
        print(f"✅ Pandas version: {pd.__version__}")
        print(f"✅ Scikit-learn version: {sklearn.__version__}")
        print(f"✅ mps available: {torch.mps.is_available()}")
        
        if torch.mps.is_available():
            print(f"✅ GPU count: {torch.mps.device_count()}")
            print(f"✅ GPU name: {torch.mps.get_device_name(0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ System info error: {e}")
        return False

def main():
    """Main test function"""
    print("🏥 ECG AI MODEL SYSTEM - TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Data Loader", test_data_loader),
        ("Utilities", test_utilities),
        ("System Info", test_system_info)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready to use.")
        print("\nTo start using the system, run:")
        print("  python main.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
