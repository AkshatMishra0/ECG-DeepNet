"""
ECG CNN Model Architecture
PyTorch implementation of CNN for ECG classification
"""

import torch
import torch.nn as nn
import numpy as np

class ECGCNN(nn.Module):
    """CNN model for ECG classification with 12-lead input"""
    
    def __init__(self, input_length=1000, num_classes=5, dropout_rate=0.5):
        super(ECGCNN, self).__init__()
        
        self.input_length = input_length
        self.num_classes = num_classes
        
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
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=15, padding=7)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(2)
        
        # Adaptive pooling to support variable input lengths
        # Compute the expected post-conv sequence length for the configured input_length
        conv_output_size = self._get_conv_output_size(input_length)
        # Each time dimension feature map has 256 channels before FC
        self._target_seq_len = max(1, conv_output_size // 256)
        self.gap = nn.AdaptiveAvgPool1d(self._target_seq_len)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate * 0.6)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def _get_conv_output_size(self, input_length):
        """Calculate the output size after convolutional layers"""
        x = torch.zeros(1, 12, input_length)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        return int(np.prod(x.size()))
    
    def forward(self, x):
        # Ensure input has correct shape (batch_size, channels, length)
        if x.dim() == 2:
            # If input is (batch_size, length), expand to (batch_size, 12, length)
            x = x.unsqueeze(1).repeat(1, 12, 1)
        elif x.dim() == 3 and x.size(1) != 12:
            # If input is (batch_size, length, channels), transpose
            x = x.transpose(1, 2)
        
        # Convolutional layers
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        # Ensure a fixed-length representation regardless of input length
        x = self.gap(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.fc3(x))
        
        return x
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_length': self.input_length,
            'num_classes': self.num_classes
        }

class ECGLSTM(nn.Module):
    """LSTM model for ECG classification (alternative architecture)"""
    
    def __init__(self, input_size=12, hidden_size=128, num_layers=2, num_classes=5, dropout_rate=0.3):
        super(ECGLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_rate, bidirectional=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 256)  # *2 for bidirectional
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate * 0.5)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        if x.dim() == 2:
            # If input is (batch_size, length), reshape to (batch_size, length, 1)
            x = x.unsqueeze(-1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.relu(self.fc1(lstm_out))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.fc3(x))
        
        return x
