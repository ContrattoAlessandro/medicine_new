import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.base_model import ClassificationModel
from models.patchtst.PatchTST import Model as PatchTST_BaseModel

class PatchTSTConfig:
    def __init__(self, enc_in, seq_len, pred_len=0, e_layers=3, n_heads=16, d_model=128,
                 d_ff=256, dropout=0.2, fc_dropout=0.0, head_dropout=0.0, individual=0,
                 patch_len=16, stride=8, padding_patch='end', revin=1, affine=1,
                 subtract_last=0, decomposition=0, kernel_size=25):
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.e_layers = e_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.head_dropout = head_dropout
        self.individual = individual
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.revin = revin
        self.affine = affine
        self.subtract_last = subtract_last
        self.decomposition = decomposition
        self.kernel_size = kernel_size

class ECGDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.y is not None:
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return x, y
        return x

class PatchTSTForClassification(nn.Module):
    def __init__(self, configs, n_classes, dropout=0.5):
        super().__init__()
        self.patchtst = PatchTST_BaseModel(configs)
        
        self.n_vars = configs.enc_in
        self.d_model = configs.d_model
        
        self.head_in_features = self.n_vars * self.d_model
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.head_in_features),
            nn.Dropout(dropout),
            nn.Linear(self.head_in_features, n_classes)
        )
        
    def forward(self, x):
        # Input shape: [bs, seq_len, channels]
        z = x.permute(0, 2, 1) # [bs, channels, seq_len]
        
        if self.patchtst.model.revin:
            z = z.permute(0, 2, 1) # [bs, seq_len, channels]
            z = self.patchtst.model.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1) # [bs, channels, seq_len]
            
        if self.patchtst.model.padding_patch == 'end':
            z = self.patchtst.model.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patchtst.model.patch_len, step=self.patchtst.model.stride) # [bs, channels, patch_num, patch_len]
        z = z.permute(0, 1, 3, 2) # [bs, channels, patch_len, patch_num]
        
        z = self.patchtst.model.backbone(z) # [bs, channels, d_model, patch_num]
        
        # Global Average Pooling (GAP) over the patch dimension (dim=-1)
        z = torch.mean(z, dim=-1) # [bs, channels, d_model]
        
        # Flatten to [bs, channels * d_model]
        z = z.reshape(z.size(0), -1)
        
        out = self.classifier(z) # [bs, n_classes]
        return out

class PatchTSTModel(ClassificationModel):
    def __init__(self, name, n_classes, freq, outputfolder, input_shape,
                 patch_len=16, stride=8, n_layers=3, d_model=128, n_heads=16,
                 d_ff=256, dropout=0.2, lr=1e-3, epochs=30, batch_size=128,
                 revin=True, device='cuda', early_stopping_patience=5):
        super().__init__()
        self.name = name
        self.n_classes = n_classes
        self.freq = freq
        self.outputfolder = outputfolder
        self.input_shape = input_shape # (seq_len, channels)
        
        self.patch_len = patch_len
        self.stride = stride
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.revin = revin
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.early_stopping_patience = early_stopping_patience
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val):
        # convert X_train/X_val and y_train/y_val to float32 numpy arrays
        X_train = np.array([l.astype(np.float32) for l in X_train])
        X_val = np.array([l.astype(np.float32) for l in X_val])
        y_train = np.array([l.astype(np.float32) for l in y_train])
        y_val = np.array([l.astype(np.float32) for l in y_val])
        
        train_dataset = ECGDataset(X_train, y_train)
        val_dataset = ECGDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        configs = PatchTSTConfig(
            enc_in=self.input_shape[1],
            seq_len=self.input_shape[0],
            pred_len=0,
            e_layers=self.n_layers,
            n_heads=self.n_heads,
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout=self.dropout,
            patch_len=self.patch_len,
            stride=self.stride,
            padding_patch='end',
            revin=1 if self.revin else 0
        )
        
        self.model = PatchTSTForClassification(configs, self.n_classes)
        self.model.to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-2)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
                
            train_loss /= len(train_dataset)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    output = self.model(batch_x)
                    loss = criterion(output, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
            val_loss /= len(val_dataset)
            
            print(f"[{self.name}] Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}", flush=True)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                os.makedirs(self.outputfolder, exist_ok=True)
                torch.save(best_model_state, os.path.join(self.outputfolder, "best_model.pt"))
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}", flush=True)
                    break
                    
        if best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})

    def predict(self, X):
        X = np.array([l.astype(np.float32) for l in X])
        
        if self.model is None:
            configs = PatchTSTConfig(
                enc_in=self.input_shape[1],
                seq_len=self.input_shape[0],
                pred_len=0,
                e_layers=self.n_layers,
                n_heads=self.n_heads,
                d_model=self.d_model,
                d_ff=self.d_ff,
                dropout=self.dropout,
                patch_len=self.patch_len,
                stride=self.stride,
                padding_patch='end',
                revin=1 if self.revin else 0
            )
            self.model = PatchTSTForClassification(configs, self.n_classes)
            model_path = os.path.join(self.outputfolder, "best_model.pt")
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            
        self.model.eval()
        dataset = ECGDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(self.device)
                output = self.model(batch_x)
                probs = torch.sigmoid(output)
                preds.append(probs.cpu().numpy())
                
        return np.concatenate(preds, axis=0)
