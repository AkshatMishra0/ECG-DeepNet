"""
ECG Model Training Script
Separate training script for ECG classification models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import time
from datetime import datetime

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from src.models.ecg_cnn import ECGCNN, ECGLSTM
from src.data.ecg_dataset import ECGDataLoader
from src.utils.model_utils import ModelSaver, TrainingMetrics

class ECGTrainer:
    """ECG Model Trainer"""
    
    def __init__(self, model_type='cnn', device=None, save_dir='./saved_models/'):
        self.model_type = model_type
        # Prefer mps, then MPS on Apple Silicon, else CPU
        if device is not None:
            self.device = device
        else:
            if torch.mps.is_available():
                self.device = torch.device('mps')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        self.save_dir = save_dir
        self.model = None
        self.data_loader = None
        self.metrics = TrainingMetrics()
        self.saver = ModelSaver(save_dir)
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Using device: {self.device}")
        print(f"Model type: {model_type.upper()}")
    
    def setup_model(self, input_length=1000, num_classes=5):
        """Initialize the model"""
        if self.model_type.lower() == 'cnn':
            self.model = ECGCNN(input_length=input_length, num_classes=num_classes)
        elif self.model_type.lower() == 'lstm':
            self.model = ECGLSTM(input_size=12, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model = self.model.to(self.device)
        
        # Print model info
        model_info = self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
        print(f"Model initialized with {model_info.get('total_parameters', 'unknown')} parameters")
    
    def setup_data(self, data_path='./DATA/', sampling_rate=100):
        """Setup data loader"""
        self.data_loader = ECGDataLoader(data_path=data_path, sampling_rate=sampling_rate)
        X, y, Y = self.data_loader.preprocess_data()
        
        # Get class names
        self.class_names = self.data_loader.get_class_names()
        num_classes = len(self.class_names)
        
        print(f"Data loaded: {X.shape[0]} samples, {num_classes} classes")
        print(f"Classes: {self.class_names}")
        
        return X, y, Y, num_classes
    
    def train(self, epochs=50, learning_rate=0.001, batch_size=32, patience=10, 
              train_ratio=0.8, test_fold=10, save_best=True):
        """Train the model"""
        print("🚀 Starting ECG Model Training")
        print("=" * 50)
        
        # Setup data
        X, y, Y, num_classes = self.setup_data()
        
        # Setup model
        self.setup_model(input_length=1000, num_classes=num_classes)
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.data_loader.create_data_loaders(
            X, y, Y, test_fold=test_fold, batch_size=batch_size, train_ratio=train_ratio
        )
        
        # Setup training
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # Training parameters
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        print(f"Training parameters:")
        print(f"  Epochs: {epochs}")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Patience: {patience}")
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
        print(f"  Test samples: {len(test_loader.dataset)}")
        print()
        
        # Training loop
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss = self._train_epoch(train_loader, criterion, optimizer)
            
            # Validation phase
            val_loss = self._validate_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Calculate metrics
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]['lr']
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Save best model
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.saver.save_model(self.model, epoch, val_loss, 
                                    class_names=self.class_names,
                                    model_type=self.model_type)
                print(f"  💾 Best model saved (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"  ⏹️  Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if save_best:
            self.model = self.saver.load_best_model(self.model)
            print("  🔄 Loaded best model for evaluation")
        
        # Final evaluation
        total_time = time.time() - start_time
        print(f"\n✅ Training completed in {total_time/60:.1f} minutes")
        
        # Evaluate on test set
        test_metrics = self.evaluate(test_loader)
        
        return {
            'model': self.model,
            'class_names': self.class_names,
            'test_metrics': test_metrics,
            'training_time': total_time
        }
    
    def _train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        print("\n📊 Evaluating model on test set...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # Convert probabilities to binary predictions
                predictions = (output > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(output.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        from sklearn.metrics import f1_score, classification_report
        
        f1_macro = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
        f1_micro = f1_score(all_targets, all_predictions, average='micro', zero_division=0)
        
        print(f"Test F1-Score (Macro): {f1_macro:.4f}")
        print(f"Test F1-Score (Micro): {f1_micro:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_targets, all_predictions, 
                                  target_names=self.class_names, zero_division=0))
        
        return {
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }

def main():
    """Main training function"""
    print("🏥 ECG AI Model Training")
    print("=" * 50)
    
    # Training configuration
    config = {
        'model_type': 'cnn',  # or 'lstm'
        'epochs': 30,
        'learning_rate': 0.001,
        'batch_size': 16,
        'patience': 10,
        'data_path': './DATA/',
        'sampling_rate': 100,
        'save_dir': './saved_models/'
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize trainer
    trainer = ECGTrainer(
        model_type=config['model_type'],
        save_dir=config['save_dir']
    )
    
    # Train the model
    results = trainer.train(
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        patience=config['patience']
    )
    
    print(f"\n🎉 Training completed successfully!")
    print(f"Model saved in: {config['save_dir']}")
    print(f"Test F1-Score: {results['test_metrics']['f1_macro']:.4f}")

if __name__ == "__main__":
    main()
