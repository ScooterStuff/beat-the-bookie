"""Neural network architectures and sklearn-compatible wrappers."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Plain feed-forward MLP with two hidden layers."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.softmax(self.fc3(x))


class DeepElasticNetMLP(nn.Module):
    """Deep MLP with explicit Elastic-Net (L1 + L2) regularisation in the loss."""

    def __init__(self, input_shape: int, classes: int, l1_lambda: float = 0.002, l2_lambda: float = 0.0005):
        super().__init__()
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, classes)
        self.act = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.softmax(self.fc4(x))

    def elastic_net_loss(self, outputs: torch.Tensor, targets: torch.Tensor, criterion) -> torch.Tensor:
        loss = criterion(outputs, targets)
        l1 = sum(torch.sum(torch.abs(p)) for p in self.parameters())
        l2 = sum(torch.sum(torch.square(p)) for p in self.parameters())
        return loss + self.l1_lambda * l1 + self.l2_lambda * l2


class GRU(nn.Module):
    """Two-layer GRU classifier."""

    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=32, batch_first=True, dropout=0.3)
        self.dropout1 = nn.Dropout(0.3)
        self.gru2 = nn.GRU(input_size=32, hidden_size=16, batch_first=True, dropout=0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.fc = nn.Linear(16, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru1(x)
        out = self.dropout1(out)
        out, _ = self.gru2(out)
        out = self.dropout2(out[:, -1, :])
        return self.softmax(self.fc(out))


class HybridGRUMLP(nn.Module):
    """Hybrid GRU (sequence branch) + MLP (current-match branch) — final model."""

    def __init__(self, seq_features: int, n_features: int, num_classes: int):
        super().__init__()
        self.gru1 = nn.GRU(input_size=seq_features, hidden_size=32, batch_first=True)
        self.gru2 = nn.GRU(input_size=32, hidden_size=64, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(16, 4), nn.ReLU(), nn.Dropout(0.1),
        )
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout_gru_final = nn.Dropout(0.2)
        self.final = nn.Linear(68, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, gru_input: torch.Tensor, mlp_input: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru1(gru_input)
        out = self.dropout1(out)
        out, _ = self.gru2(out)
        out = self.dropout2(out[:, -1, :])
        out = self.dropout_gru_final(out)
        mlp_out = self.mlp(mlp_input)
        return self.softmax(self.final(torch.cat((out, mlp_out), dim=1)))


# ---------------------------------------------------------------------------
# Sklearn-compatible adaptors (used as base learners by the stacking model)
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SklearnCompatibleGRUMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, num_epochs: int = 10, batch_size: int = 32, learning_rate: float = 3e-4, device: str | None = None):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        _set_seed(50)
        self.model = HybridGRUMLP(seq_features=X.shape[1], n_features=X.shape[1], num_classes=len(set(y))).to(self.device)

        X_gru = torch.FloatTensor(X.reshape((X.shape[0], 1, X.shape[1]))).to(self.device)
        X_mlp = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y.values if isinstance(y, pd.Series) else y).to(self.device)

        loader = DataLoader(TensorDataset(X_gru, X_mlp, y_t), batch_size=self.batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            preds, labels = [], []
            for bg, bm, by in loader:
                optimizer.zero_grad()
                out = self.model(bg, bm)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                preds.extend(torch.max(out.data, 1)[1].cpu().numpy())
                labels.extend(by.cpu().numpy())
            if (epoch + 1) % max(1, self.num_epochs // 5) == 0:
                print(f"[GRU+MLP] epoch {epoch+1}/{self.num_epochs} loss={total_loss/len(loader):.4f} acc={accuracy_score(labels, preds):.4f}")

        self.classes_ = torch.unique(y_t).cpu().numpy()
        return self

    def _to_inputs(self, X):
        return (
            torch.FloatTensor(X.reshape((X.shape[0], 1, X.shape[1]))).to(self.device),
            torch.FloatTensor(X).to(self.device),
        )

    def predict(self, X):
        Xg, Xm = self._to_inputs(X)
        self.model.eval()
        with torch.no_grad():
            return torch.argmax(self.model(Xg, Xm), dim=1).cpu().numpy()

    def predict_proba(self, X):
        Xg, Xm = self._to_inputs(X)
        self.model.eval()
        with torch.no_grad():
            return F.softmax(self.model(Xg, Xm), dim=1).cpu().numpy()


class SklearnCompatibleElasticNetMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, num_epochs: int = 50, batch_size: int = 32, learning_rate: float = 1e-3):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X, y):
        _set_seed(90)
        self.model = DeepElasticNetMLP(X.shape[1], len(set(y))).to(self.device)
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y.values if isinstance(y, pd.Series) else y).to(self.device)

        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.model(bx)
                loss = self.model.elastic_net_loss(out, by, criterion)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 20 == 0:
                print(f"[ElasticNet] epoch {epoch+1}/{self.num_epochs} loss={total_loss/len(loader):.4f}")

        self.classes_ = np.unique(y_t.cpu().numpy())
        return self

    def predict(self, X):
        X_t = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(X_t).argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        X_t = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return F.softmax(self.model(X_t), dim=1).cpu().numpy()
