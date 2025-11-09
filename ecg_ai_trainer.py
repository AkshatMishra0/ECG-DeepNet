import pandas as pd
import numpy as np
import wfdb
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import joblib
import os
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ECGDataset(Dataset):
    """PyTorch Dataset for ECG data"""
    def __init__(self, signals, labels, transform=None):
        self.signals = signals
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = torch.FloatTensor(self.signals[idx])
        label = torch.FloatTensor(self.labels[idx])
        
        if self.transform:
            signal = self.transform(signal)
        
        return signal, label

class ECGCNN(nn.Module):
    """CNN model for ECG classification"""
    def __init__(self, input_length=1000, num_classes=5):
        super(ECGCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(12, 32, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=15, padding=7)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=15, padding=7)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate the size after convolutions
        conv_output_size = self._get_conv_output_size(input_length)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def _get_conv_output_size(self, input_length):
        """Calculate the output size after convolutional layers"""
        x = torch.zeros(1, 12, input_length)
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        return int(np.prod(x.size()))
    
    def forward(self, x):
        # Reshape input to (batch_size, channels, length)
        if x.dim() == 2:
            x = x.unsqueeze(1).repeat(1, 12, 1)
        
        # Convolutional layers
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.fc3(x))
        
        return x

class ECGAITrainer:
    """AI Model Trainer for ECG Classification"""
    
    def __init__(self, data_path='./DATA/', sampling_rate=100, device=None):
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        self.device = device if device else torch.device('mps' if torch.mps.is_available() else 'cpu')
        self.model = None
        self.label_encoder = None
        self.scaler = None
        self.class_names = None
        
    def load_raw_data(self, df, sampling_rate, path):
        """Load raw ECG data from files"""
        print("Loading ECG data...")
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data
    
    def preprocess_data(self):
        """Load and preprocess the ECG dataset"""
        print("Preprocessing data...")
        
        # Load annotation data
        Y = pd.read_csv(self.data_path + 'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        
        # Load raw signal data
        X = self.load_raw_data(Y, self.sampling_rate, self.data_path)
        
        # Load diagnostic statements
        agg_df = pd.read_csv(self.data_path + 'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
        
        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))
        
        # Apply diagnostic superclass
        Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
        
        # Filter out empty diagnostic classes
        Y = Y[Y.diagnostic_superclass.apply(lambda x: len(x) > 0)]
        
        # Get unique classes
        all_classes = []
        for classes in Y.diagnostic_superclass:
            all_classes.extend(classes)
        self.class_names = list(set(all_classes))
        
        # Encode labels
        self.label_encoder = MultiLabelBinarizer()
        y_encoded = self.label_encoder.fit_transform(Y.diagnostic_superclass)
        
        # Get corresponding X data
        X_filtered = X[Y.index]
        
        return X_filtered, y_encoded, Y
    
    def create_data_loaders(self, X, y, test_fold=10, batch_size=32, train_ratio=0.8):
        """Create train and test data loaders"""
        print("Creating data loaders...")
        
        # Load metadata for fold splitting
        Y = pd.read_csv(self.data_path + 'ptbxl_database.csv', index_col='ecg_id')
        
        # Filter to match our data
        Y_filtered = Y.loc[X.index] if hasattr(X, 'index') else Y.iloc[:len(X)]
        
        # Split data
        train_mask = Y_filtered.strat_fold != test_fold
        test_mask = Y_filtered.strat_fold == test_fold
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Further split training data into train/validation
        n_train = int(len(X_train) * train_ratio)
        indices = np.random.permutation(len(X_train))
        
        X_train_split = X_train[indices[:n_train]]
        X_val_split = X_train[indices[n_train:]]
        y_train_split = y_train[indices[:n_train]]
        y_val_split = y_train[indices[n_train:]]
        
        # Create datasets
        train_dataset = ECGDataset(X_train_split, y_train_split)
        val_dataset = ECGDataset(X_val_split, y_val_split)
        test_dataset = ECGDataset(X_test, y_test)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, epochs=50, learning_rate=0.001, batch_size=32, patience=10):
        """Train the ECG classification model"""
        print("Starting model training...")
        
        # Preprocess data
        X, y, Y = self.preprocess_data()
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders(X, y, batch_size=batch_size)
        
        # Initialize model
        num_classes = len(self.class_names)
        self.model = ECGCNN(input_length=1000, num_classes=num_classes).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    val_loss += criterion(output, target).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_ecg_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_ecg_model.pth'))
        
        # Evaluate on test set
        self.evaluate_model(test_loader)
        
        print("Training completed!")
        return self.model
    
    def evaluate_model(self, test_loader):
        """Evaluate the model on test data"""
        print("Evaluating model...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # Convert probabilities to binary predictions
                predictions = (output > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        f1_macro = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
        f1_micro = f1_score(all_targets, all_predictions, average='micro', zero_division=0)
        
        print(f"Test F1-Score (Macro): {f1_macro:.4f}")
        print(f"Test F1-Score (Micro): {f1_micro:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_targets, all_predictions, 
                                  target_names=self.class_names, zero_division=0))
        
        return f1_macro, f1_micro
    
    def predict(self, new_data):
        """Make predictions on new ECG data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        self.model.eval()
        
        # Preprocess new data if needed
        if isinstance(new_data, np.ndarray):
            if new_data.ndim == 2:
                new_data = new_data.reshape(1, -1)
        else:
            new_data = np.array(new_data)
        
        # Convert to tensor
        new_data = torch.FloatTensor(new_data).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(new_data)
            predictions = (predictions > 0.5).float()
        
        # Convert to class names
        predicted_classes = []
        for pred in predictions.cpu().numpy():
            classes = [self.class_names[i] for i, p in enumerate(pred) if p > 0.5]
            predicted_classes.append(classes)
        
        return predicted_classes, predictions.cpu().numpy()
    
    def save_model(self, filepath='ecg_model.pkl'):
        """Save the trained model and preprocessing objects"""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'label_encoder': self.label_encoder,
            'sampling_rate': self.sampling_rate
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='ecg_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        self.class_names = model_data['class_names']
        self.label_encoder = model_data['label_encoder']
        self.sampling_rate = model_data['sampling_rate']
        
        # Initialize and load model
        num_classes = len(self.class_names)
        self.model = ECGCNN(input_length=1000, num_classes=num_classes).to(self.device)
        self.model.load_state_dict(model_data['model_state_dict'])
        
        print(f"Model loaded from {filepath}")
        return self.model

def main():
    """Example usage of the ECG AI Trainer"""
    print("ECG AI Model Trainer")
    print("=" * 50)
    
    # Initialize trainer
    trainer = ECGAITrainer(data_path='./DATA/', sampling_rate=100)
    
    # Train the model
    model = trainer.train_model(epochs=20, learning_rate=0.001, batch_size=16)
    
    # Save the model
    trainer.save_model('ecg_classifier.pkl')
    
    # Example of making predictions on new data
    print("\nExample: Making predictions on new data...")
    
    # Load some test data for demonstration
    Y = pd.read_csv('./DATA/ptbxl_database.csv', index_col='ecg_id')
    sample_data = trainer.load_raw_data(Y.iloc[:5], 100, './DATA/')
    
    # Make predictions
    predictions, probabilities = trainer.predict(sample_data[0])
    print(f"Predicted classes: {predictions[0]}")
    print(f"Probabilities: {probabilities[0]}")

if __name__ == "__main__":
    main()
