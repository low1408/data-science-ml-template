from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_is_fitted

# Keep PyTorch imports conditional at the module level to allow import when torch is not installed
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


if HAS_TORCH:
    class PyTorchTabularTransformer(nn.Module):
        """
        Internal PyTorch neural network module for a simple numeric-feature Tabular Transformer.
        """
        def __init__(
            self,
            n_features: int,
            d_model: int,
            n_heads: int,
            n_layers: int,
            dim_feedforward: int,
            dropout: float,
            n_classes: int,
            generator: torch.Generator | None = None,
        ):
            super().__init__()
            # Learnable scale and bias per numeric feature
            if generator is not None:
                self.feature_weight = nn.Parameter(
                    torch.empty(n_features, d_model).normal_(mean=0.0, std=0.02, generator=generator)
                )
                self.feature_bias = nn.Parameter(torch.zeros(n_features, d_model))
                self.cls_token = nn.Parameter(
                    torch.empty(1, 1, d_model).normal_(mean=0.0, std=0.02, generator=generator)
                )
            else:
                self.feature_weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
                self.feature_bias = nn.Parameter(torch.zeros(n_features, d_model))
                self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            
            # Project CLS token representation to classification logits
            self.fc = nn.Linear(d_model, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: [batch, n_features]
            # Project continuous features to [batch, n_features, d_model]
            tokens = x.unsqueeze(-1) * self.feature_weight + self.feature_bias
            
            # Prepend [CLS] token: [batch, 1, d_model]
            batch_size = x.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)  # [batch, n_features + 1, d_model]
            
            # Forward pass through transformer encoder
            encoded = self.transformer(tokens)  # [batch, n_features + 1, d_model]
            
            # Pull out the encoded [CLS] token (index 0) and project
            logits = self.fc(encoded[:, 0])  # [batch, n_classes]
            return logits
else:
    # Fallback to prevent syntax errors when defining subclasses if torch is not installed
    class PyTorchTabularTransformer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass


class TabularTransformerClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-Learn compatible PyTorch Tabular Transformer classifier (Vanilla variant).
    """
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 10,
        weight_decay: float = 0.0,
        random_state: int | None = None,
        device: str = "cpu",
        verbose: bool = False,
    ):
        # Strict contract: do not instantiate the PyTorch network graph or state here.
        # Assign constructor parameters to attributes verbatim.
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def fit(self, X, y) -> TabularTransformerClassifier:
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required to train and run TabularTransformerClassifier. "
                "Please run `pip install torch` in your environment."
            )

        # Densify sparse input locally if needed (e.g. scipy sparse matrix)
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Validate inputs
        X, y = check_X_y(X, y, accept_sparse=False)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]

        if len(self.classes_) < 2:
            raise ValueError("The target y must contain at least 2 distinct classes.")

        # Save global RNG state for safety
        torch_rng = torch.get_rng_state()
        np_rng = np.random.get_state()

        try:
            # Multi-level random seeding for strict reproducibility
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.random_state)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            # Validate transformer parameters
            if self.d_model % self.n_heads != 0:
                raise ValueError(
                    f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})."
                )

            # Instantiate network graph and move it to the device here (in fit, not __init__)
            self.model_ = PyTorchTabularTransformer(
                n_features=self.n_features_in_,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                n_classes=len(self.classes_),
            ).to(self.device)

            # Map target labels to contiguous integer indices: 0 to (n_classes - 1)
            # Unique classes_ contains original sorted labels. Get mapping index.
            label_to_index = {label: idx for idx, label in enumerate(self.classes_)}
            y_encoded = np.array([label_to_index[val] for val in y], dtype=np.int64)

            # Prepare PyTorch Dataloader
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y_encoded, dtype=torch.long)
            dataset = TensorDataset(X_tensor, y_tensor)
            
            # Use drop_last=False to guarantee all samples are trained
            loader = DataLoader(
                dataset,
                batch_size=min(self.batch_size, len(X)),
                shuffle=True,
                drop_last=False,
            )

            optimizer = optim.AdamW(
                self.model_.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            criterion = nn.CrossEntropyLoss()

            self.model_.train()
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for batch_X, batch_y in loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()
                    logits = self.model_(batch_X)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * len(batch_X)
                
                if self.verbose:
                    print(f"Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss / len(X):.4f}")
        finally:
            # Restore RNG state
            torch.set_rng_state(torch_rng)
            np.random.set_state(np_rng)

        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        
        if hasattr(X, "toarray"):
            X = X.toarray()
            
        # Ensure array type matches n_features_in_
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_features_in_} features, "
                f"but input has {X_arr.shape[1]} features."
            )

        self.model_.eval()
        with torch.no_grad():
            t_X = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
            logits = self.model_(t_X)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X) -> np.ndarray:
        probas = self.predict_proba(X)
        class_indices = np.argmax(probas, axis=1)
        return self.classes_[class_indices]


def _focal_loss(
    logits,
    targets,
    gamma: float,
    class_weights = None,
    sample_weights = None,
):
    """
    Compute multi-class Focal Loss directly on logits.
    """
    # Numerically stable CE loss per sample (unweighted, for pt calculation)
    ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce_loss)
    
    # Calculate focal loss
    per_sample = ((1.0 - pt) ** gamma) * ce_loss

    if class_weights is not None:
        per_sample = per_sample * class_weights[targets]

    if sample_weights is not None:
        per_sample = per_sample * sample_weights
        return per_sample.sum() / sample_weights.sum().clamp_min(1e-12)

    return per_sample.mean()


class AdvancedTabularTransformerClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-Learn compatible PyTorch Tabular Transformer classifier (Advanced variant).
    Supports class/sample weights, weighted random sampling, focal loss, early stopping
    with internal train/val splitting, schedulers, parameter groupings, and local generators.
    """
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 10,
        weight_decay: float = 0.0,
        random_state: int | None = None,
        device: str = "cpu",
        verbose: bool = False,
        class_weight: str | dict | None = None,
        loss: str = "cross_entropy",
        focal_gamma: float = 2.0,
        sampling_strategy: str = "none",
        lr_scheduler: str | None = None,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        scheduler_step_size: int = 3,
        scheduler_gamma: float = 0.1,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        gradient_clip_norm: float | None = None,
        exclude_decay_on_bias_norm: bool = True,
    ):
        # Strict contract: assign constructor parameters to attributes verbatim.
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device
        self.verbose = verbose
        
        self.class_weight = class_weight
        self.loss = loss
        self.focal_gamma = focal_gamma
        self.sampling_strategy = sampling_strategy
        self.lr_scheduler = lr_scheduler
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.gradient_clip_norm = gradient_clip_norm
        self.exclude_decay_on_bias_norm = exclude_decay_on_bias_norm

    def fit(self, X, y, sample_weight: np.ndarray | None = None) -> AdvancedTabularTransformerClassifier:
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required to train and run AdvancedTabularTransformerClassifier. "
                "Please run `pip install torch` in your environment."
            )

        # Save global RNG states to guarantee 100% side-effect-free execution
        torch_rng_state = torch.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        numpy_rng_state = np.random.get_state()

        try:
            # 1. Validate parameter bounds
            if self.focal_gamma < 0:
                raise ValueError(f"focal_gamma must be >= 0, got {self.focal_gamma}.")
            if self.patience < 1:
                raise ValueError(f"patience must be >= 1, got {self.patience}.")
            if not (0.0 < self.validation_fraction < 1.0):
                raise ValueError(f"validation_fraction must be in (0, 1), got {self.validation_fraction}.")
            if not (0.0 < self.scheduler_factor < 1.0):
                raise ValueError(f"scheduler_factor must be in (0, 1), got {self.scheduler_factor}.")
            if self.scheduler_step_size < 1:
                raise ValueError(f"scheduler_step_size must be >= 1, got {self.scheduler_step_size}.")
            if self.loss not in ["cross_entropy", "crossentropy", "focal"]:
                raise ValueError(f"Unsupported loss {self.loss!r}. Must be 'cross_entropy', 'crossentropy' or 'focal'.")
            if self.sampling_strategy not in ["none", "weighted"]:
                raise ValueError(f"Unsupported sampling_strategy {self.sampling_strategy!r}.")
            if self.lr_scheduler not in [None, "cosine", "step", "plateau"]:
                raise ValueError(f"Unsupported lr_scheduler {self.lr_scheduler!r}.")

            # Densify sparse input locally if needed (e.g. scipy sparse matrix)
            if hasattr(X, "toarray"):
                X = X.toarray()

            # Validate inputs
            X, y = check_X_y(X, y, accept_sparse=False)
            self.classes_ = unique_labels(y)
            self.n_features_in_ = X.shape[1]

            n_classes = len(self.classes_)
            if n_classes < 2:
                raise ValueError("The target y must contain at least 2 distinct classes.")

            if self.d_model % self.n_heads != 0:
                raise ValueError(
                    f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})."
                )

            # Local generator for strict reproducibility without mutating global states
            generator = torch.Generator()
            if self.random_state is not None:
                generator.manual_seed(self.random_state)

            # Map target labels to contiguous integer indices: 0 to (n_classes - 1)
            label_to_index = {label: idx for idx, label in enumerate(self.classes_)}
            y_encoded = np.array([label_to_index[val] for val in y], dtype=np.int64)

            # Align and validate sample weights
            if sample_weight is not None:
                sample_weight = np.asarray(sample_weight, dtype=np.float32)
                if len(sample_weight) != len(y):
                    raise ValueError(
                        f"Length of sample_weight ({len(sample_weight)}) must match y ({len(y)})."
                    )
                if np.any(sample_weight < 0):
                    raise ValueError("sample_weight elements must be non-negative.")
            else:
                sample_weight = np.ones(len(y), dtype=np.float32)

            # 2. Compute class weights (saved to self.class_weight_ to avoid shadowing parameter)
            if self.class_weight is None:
                self.class_weight_ = None
            elif self.class_weight == "balanced":
                counts = np.bincount(y_encoded, minlength=n_classes)
                # Prevent division by zero
                counts = np.clip(counts, a_min=1, a_max=None)
                self.class_weight_ = len(y_encoded) / (n_classes * counts)
            elif isinstance(self.class_weight, dict):
                missing = [label for label in self.classes_ if label not in self.class_weight]
                if missing:
                    raise ValueError(f"Missing weights for classes: {missing}")
                self.class_weight_ = np.asarray(
                    [self.class_weight[label] for label in self.classes_],
                    dtype=np.float32,
                )
            else:
                raise ValueError("class_weight must be None, 'balanced', or a mapping.")

            class_weight_tensor = (
                None if self.class_weight_ is None
                else torch.as_tensor(self.class_weight_, dtype=torch.float32, device=self.device)
            )

            # 3. Handle validation splitting
            val_loader = None
            if self.early_stopping or self.lr_scheduler == "plateau":
                # Stratified train/val split using sklearn model_selection helper
                try:
                    X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
                        X,
                        y_encoded,
                        sample_weight,
                        test_size=self.validation_fraction,
                        stratify=y_encoded,
                        random_state=self.random_state,
                    )
                except ValueError as split_error:
                    # Fallback to non-stratified split if stratification fails due to small class counts
                    X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
                        X,
                        y_encoded,
                        sample_weight,
                        test_size=self.validation_fraction,
                        random_state=self.random_state,
                    )
                
                val_dataset = TensorDataset(
                    torch.tensor(X_val, dtype=torch.float32),
                    torch.tensor(y_val, dtype=torch.long),
                    torch.tensor(sw_val, dtype=torch.float32),
                )
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            else:
                X_train, y_train, sw_train = X, y_encoded, sample_weight

            train_dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
                torch.tensor(sw_train, dtype=torch.float32),
            )

            # 4. Handle weighted sampling sampler construction
            sampler = None
            if self.sampling_strategy == "weighted":
                counts = np.bincount(y_train, minlength=n_classes)
                # Avoid division by zero
                counts = np.clip(counts, a_min=1, a_max=None)
                inverse_frequency = 1.0 / counts
                per_sample_sampling_weight = inverse_frequency[y_train]
                sampler = WeightedRandomSampler(
                    weights=torch.as_tensor(per_sample_sampling_weight, dtype=torch.double),
                    num_samples=len(train_dataset),
                    replacement=True,
                    generator=generator,
                )

            train_loader = DataLoader(
                train_dataset,
                batch_size=min(self.batch_size, len(train_dataset)),
                shuffle=sampler is None,
                sampler=sampler,
                drop_last=False,
                generator=generator,
            )

            # 5. Instantiate model graph
            self.model_ = PyTorchTabularTransformer(
                n_features=self.n_features_in_,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                n_classes=n_classes,
                generator=generator,
            ).to(self.device)

            # 6. Apply parameter grouping (exclude decay on bias / normalization layers)
            if self.exclude_decay_on_bias_norm and self.weight_decay > 0.0:
                decay_params = []
                no_decay_params = []
                for name, param in self.model_.named_parameters():
                    if not param.requires_grad:
                        continue
                    lowered_name = name.lower()
                    if param.ndim == 1 or lowered_name.endswith("bias") or "norm" in lowered_name:
                        no_decay_params.append(param)
                    else:
                        decay_params.append(param)

                optimizer = optim.AdamW(
                    [
                        {"params": decay_params, "weight_decay": self.weight_decay},
                        {"params": no_decay_params, "weight_decay": 0.0},
                    ],
                    lr=self.learning_rate,
                )
            else:
                optimizer = optim.AdamW(
                    self.model_.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )

            # 7. Setup Loss Criterion
            if self.loss in ["cross_entropy", "crossentropy"]:
                criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, reduction="none")

            # 8. Setup Scheduler
            scheduler = None
            if self.lr_scheduler == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
            elif self.lr_scheduler == "step":
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma
                )
            elif self.lr_scheduler == "plateau":
                if val_loader is None:
                    raise ValueError("Plateau LR scheduler requires validation splitting or early_stopping.")
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", patience=self.scheduler_patience, factor=self.scheduler_factor
                )

            # 9. Training loop
            self.history_ = {"train_loss": [], "val_loss": []}
            best_val_loss = np.inf
            best_state = None
            bad_epochs = 0
            self.n_iter_ = 0

            for epoch in range(self.epochs):
                self.model_.train()
                train_loss = 0.0
                total_samples = 0

                for batch_X, batch_y, batch_sw in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    batch_sw = batch_sw.to(self.device)

                    optimizer.zero_grad()
                    logits = self.model_(batch_X)

                    # Compute loss based on loss type
                    if self.loss in ["cross_entropy", "crossentropy"]:
                        per_sample_loss = criterion(logits, batch_y)
                        # Average over batch weighted by sample weight
                        loss = (per_sample_loss * batch_sw).sum() / batch_sw.sum().clamp_min(1e-12)
                    else:  # Focal loss
                        loss = _focal_loss(
                            logits=logits,
                            targets=batch_y,
                            gamma=self.focal_gamma,
                            class_weights=class_weight_tensor,
                            sample_weights=batch_sw,
                        )

                    loss.backward()
                    
                    # Apply gradient clipping if requested
                    if self.gradient_clip_norm is not None:
                        nn.utils.clip_grad_norm_(self.model_.parameters(), self.gradient_clip_norm)

                    optimizer.step()

                    train_loss += loss.item() * len(batch_X)
                    total_samples += len(batch_X)

                self.n_iter_ += 1
                epoch_train_loss = train_loss / total_samples
                self.history_["train_loss"].append(epoch_train_loss)

                # Evaluate validation loss
                epoch_val_loss = None
                if val_loader is not None:
                    self.model_.eval()
                    val_loss_sum = 0.0
                    val_samples = 0
                    with torch.no_grad():
                        for batch_X, batch_y, batch_sw in val_loader:
                            batch_X = batch_X.to(self.device)
                            batch_y = batch_y.to(self.device)
                            batch_sw = batch_sw.to(self.device)
                            logits = self.model_(batch_X)

                            if self.loss in ["cross_entropy", "crossentropy"]:
                                per_sample_loss = criterion(logits, batch_y)
                                loss = (per_sample_loss * batch_sw).sum() / batch_sw.sum().clamp_min(1e-12)
                            else:
                                loss = _focal_loss(
                                    logits=logits,
                                    targets=batch_y,
                                    gamma=self.focal_gamma,
                                    class_weights=class_weight_tensor,
                                    sample_weights=batch_sw,
                                )
                            val_loss_sum += loss.item() * len(batch_X)
                            val_samples += len(batch_X)
                    epoch_val_loss = val_loss_sum / val_samples
                    self.history_["val_loss"].append(epoch_val_loss)

                if self.verbose:
                    val_suffix = f" - Val Loss: {epoch_val_loss:.4f}" if epoch_val_loss is not None else ""
                    print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {epoch_train_loss:.4f}{val_suffix}")

                # Early stopping and checkpointing logic
                if self.early_stopping and epoch_val_loss is not None:
                    if epoch_val_loss < best_val_loss - self.min_delta:
                        best_val_loss = epoch_val_loss
                        best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                        bad_epochs = 0
                    else:
                        bad_epochs += 1

                    if bad_epochs >= self.patience:
                        if self.verbose:
                            print(f"Early stopping triggered at epoch {epoch+1}.")
                        break

                # Step scheduler
                if scheduler is not None:
                    if self.lr_scheduler == "plateau":
                        # ReduceLROnPlateau expects validation metric
                        scheduler.step(epoch_val_loss if epoch_val_loss is not None else epoch_train_loss)
                    else:
                        scheduler.step()

            # Restore best weights if early stopping was active
            if self.early_stopping and self.restore_best_weights and best_state is not None:
                self.model_.load_state_dict(best_state)

        finally:
            # Restore global RNG states
            torch.set_rng_state(torch_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
            np.random.set_state(numpy_rng_state)

        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        
        if hasattr(X, "toarray"):
            X = X.toarray()
            
        # Ensure array type matches n_features_in_
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_features_in_} features, "
                f"but input has {X_arr.shape[1]} features."
            )

        self.model_.eval()
        with torch.no_grad():
            t_X = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
            logits = self.model_(t_X)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X) -> np.ndarray:
        probas = self.predict_proba(X)
        class_indices = np.argmax(probas, axis=1)
        return self.classes_[class_indices]
