"""
Model Utilities
Utilities for saving, loading, and managing PyTorch models
"""

import torch
import os
import json
from datetime import datetime
import numpy as np

class ModelSaver:
    """Utility class for saving and loading PyTorch models"""
    
    def __init__(self, save_dir='./saved_models/'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_model(self, model, epoch, val_loss, class_names=None, model_type='cnn', 
                   optimizer_state=None, scheduler_state=None, additional_info=None):
        """Save model with metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"ecg_model_{model_type}_{timestamp}.pth"
        filepath = os.path.join(self.save_dir, filename)
        
        # Prepare model data
        model_data = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'class_names': class_names,
            'model_type': model_type,
            'timestamp': timestamp,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler_state,
            'additional_info': additional_info or {}
        }
        
        # Save model
        torch.save(model_data, filepath)
        
        # Also save as 'best_model.pth' for easy loading
        best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        torch.save(model_data, best_model_path)
        
        print(f"Model saved: {filepath}")
        return filepath
    
    def load_model(self, filepath, model=None, device=None):
        """Load model from file"""
        if device is None:
            device = torch.device('mps' if torch.mps.is_available() else 'cpu')
        
        print(f"Loading model from: {filepath}")
        
        try:
            model_data = torch.load(filepath, map_location=device)
            
            if model is not None:
                model.load_state_dict(model_data['model_state_dict'])
                model = model.to(device)
            
            return model, model_data
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_best_model(self, model, device=None):
        """Load the best model"""
        best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        
        if not os.path.exists(best_model_path):
            raise FileNotFoundError("No best model found")
        
        model, model_data = self.load_model(best_model_path, model, device)
        return model
    
    def list_models(self):
        """List all saved models"""
        models = []
        for filename in os.listdir(self.save_dir):
            if filename.endswith('.pth'):
                filepath = os.path.join(self.save_dir, filename)
                try:
                    model_data = torch.load(filepath, map_location='cpu')
                    models.append({
                        'filename': filename,
                        'filepath': filepath,
                        'epoch': model_data.get('epoch', 'unknown'),
                        'val_loss': model_data.get('val_loss', 'unknown'),
                        'model_type': model_data.get('model_type', 'unknown'),
                        'timestamp': model_data.get('timestamp', 'unknown'),
                        'class_names': model_data.get('class_names', [])
                    })
                except:
                    models.append({
                        'filename': filename,
                        'filepath': filepath,
                        'error': 'Could not load metadata'
                    })
        
        return models

class TrainingMetrics:
    """Class for tracking training metrics"""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.epochs = []
    
    def update(self, epoch, train_loss, val_loss, lr):
        """Update metrics"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)
    
    def get_best_epoch(self):
        """Get epoch with best validation loss"""
        if not self.val_losses:
            return None
        best_idx = np.argmin(self.val_losses)
        return self.epochs[best_idx], self.val_losses[best_idx]
    
    def plot_metrics(self, save_path=None):
        """Plot training metrics"""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot losses
            ax1.plot(self.epochs, self.train_losses, label='Train Loss', color='blue')
            ax1.plot(self.epochs, self.val_losses, label='Val Loss', color='red')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot learning rate
            ax2.plot(self.epochs, self.learning_rates, label='Learning Rate', color='green')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Metrics plot saved to: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib not available for plotting")

class ModelEvaluator:
    """Class for evaluating model performance"""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device else torch.device('mps' if torch.mps.is_available() else 'cpu')
    
    def evaluate_metrics(self, data_loader, class_names=None):
        """Evaluate model and return comprehensive metrics"""
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in data_loader:
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
        metrics = {
            'f1_macro': f1_score(all_targets, all_predictions, average='macro', zero_division=0),
            'f1_micro': f1_score(all_targets, all_predictions, average='micro', zero_division=0),
            'precision_macro': precision_score(all_targets, all_predictions, average='macro', zero_division=0),
            'recall_macro': recall_score(all_targets, all_predictions, average='macro', zero_division=0),
            'num_samples': len(all_targets),
            'num_classes': len(class_names) if class_names else all_targets.shape[1]
        }
        
        # Per-class metrics
        if class_names:
            per_class_metrics = {}
            for i, class_name in enumerate(class_names):
                per_class_metrics[class_name] = {
                    'f1': f1_score(all_targets[:, i], all_predictions[:, i], zero_division=0),
                    'precision': precision_score(all_targets[:, i], all_predictions[:, i], zero_division=0),
                    'recall': recall_score(all_targets[:, i], all_predictions[:, i], zero_division=0)
                }
            metrics['per_class'] = per_class_metrics
        
        return metrics, all_predictions, all_targets, all_probabilities
    
    def print_evaluation_report(self, metrics, class_names=None):
        """Print a comprehensive evaluation report"""
        print("\n📊 Model Evaluation Report")
        print("=" * 50)
        print(f"Number of samples: {metrics['num_samples']}")
        print(f"Number of classes: {metrics['num_classes']}")
        print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"F1-Score (Micro): {metrics['f1_micro']:.4f}")
        print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
        
        if 'per_class' in metrics and class_names:
            print("\nPer-class metrics:")
            print("-" * 30)
            for class_name, class_metrics in metrics['per_class'].items():
                print(f"{class_name:15s}: "
                      f"F1={class_metrics['f1']:.3f}, "
                      f"P={class_metrics['precision']:.3f}, "
                      f"R={class_metrics['recall']:.3f}")

def create_model_summary(model, input_shape=(1, 12, 1000)):
    """Create a summary of the model architecture"""
    try:
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_type': type(model).__name__,
            'input_shape': input_shape
        }
        
        return summary
        
    except Exception as e:
        return {'error': str(e)}
