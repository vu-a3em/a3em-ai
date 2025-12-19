"""Shared training utilities for few-shot learning and transfer learning."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
import numpy as np
from .. import keras, PyDataset


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 10000
    patience: int = 10
    train_split: float = 0.8
    dropout: float = 0.25
    hidden_units: List[int] = field(default_factory=lambda: [256, 128])
    activation: str = 'relu'
    lr_schedule: Optional[Callable] = None
    target_fpr: float = 0.10  # For None-bias threshold computation
    
    def create_lr_schedule(self, schedule_type: Optional[str] = None, **kwargs):
        """Create a learning rate schedule callback.
        
        Args:
            schedule_type: 'plateau' or 'cosine', or None for constant LR
            **kwargs: Additional parameters for the schedule
        """
        if schedule_type == 'plateau':
            return keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('schedule_patience', 5),
                min_lr=kwargs.get('min_lr', 1e-7),
                verbose=1
            )
        elif schedule_type == 'cosine':
            return keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=kwargs.get('decay_steps', 3000),
                alpha=kwargs.get('alpha', 1e-2)
            )
        return None


def generate_embedding_dataset(seq: PyDataset, model: keras.Model) -> Tuple[np.ndarray, np.ndarray]:
    """Extract embeddings from a dataset using a feature extraction model.
    
    Args:
        seq: KerasDataSet (keras.utils.Sequence) providing batches of (waveforms, labels)
        model: Feature extraction model (e.g., base model without classification head)
    
    Returns:
        X: Embeddings as numpy array of shape (num_samples, embedding_dim)
        y: Labels as numpy array of shape (num_samples,)
    """
    embs = []
    lbls = []
    for i in range(len(seq)):
        batch_waveforms, batch_labels = seq[i]
        batch_embs = model(batch_waveforms, training=False)
        # Convert to numpy if it's a Tensor
        if hasattr(batch_embs, 'numpy'):
            batch_embs = batch_embs.numpy()
        embs.append(batch_embs)
        lbls.append(batch_labels)
    
    if len(embs) == 0:
        return np.array([]), np.array([])
    
    X = np.concatenate(embs, axis=0)
    y = np.concatenate(lbls, axis=0)
    return X, y


def build_classifier_head(input_shape: Tuple[int, ...], 
                         num_classes: int, 
                         config: TrainingConfig) -> keras.Model:
    """Build a classifier head for few-shot or transfer learning.
    
    Args:
        input_shape: Shape of input embeddings (e.g., (256,))
        num_classes: Number of output classes
        config: TrainingConfig with architecture parameters
    
    Returns:
        Compiled Sequential classifier model
    """
    layers = [keras.layers.Input(shape=input_shape, dtype='float32')]
    
    for units in config.hidden_units:
        layers.append(keras.layers.Dense(units=units, activation=config.activation, use_bias=True))
        layers.append(keras.layers.Dropout(config.dropout))
    
    # Output layer
    layers.append(keras.layers.Dense(units=num_classes, use_bias=True, activation='softmax'))
    
    classifier = keras.Sequential(layers)
    return classifier


def compute_none_bias_threshold(model: keras.Model, 
                                val_ds, 
                                none_idx: int, 
                                target_fpr: float = 0.10) -> Tuple[float, List[Tuple[float, float, float]]]:
    """Compute a decision threshold to bias away from 'None' class.
    
    This computes a threshold on (best_non_none_prob - none_prob) to control
    the false positive rate of misclassifying None as an event.
    
    Args:
        model: Trained classifier (softmax output)
        val_ds: Validation dataset (tf.data.Dataset or similar)
        none_idx: Integer index of the None/background class
        target_fpr: Maximum allowed None->event false positive rate
    
    Returns:
        best_threshold: Threshold value for decision rule
        roc_points: List of (threshold, tpr, fpr) tuples for ROC curve
    """
    y_true_list, y_prob_list = [], []
    for xb, yb in val_ds:
        probs = model(xb, training=False).numpy()
        y_true_list.append(yb.numpy() if hasattr(yb, 'numpy') else yb)
        y_prob_list.append(probs)
    
    if not y_true_list:
        return 0.0, []
    
    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)
    
    num_classes = y_prob.shape[1]
    non_none_mask = np.ones(num_classes, dtype=bool)
    non_none_mask[none_idx] = False
    
    best_non_none_idx = np.argmax(y_prob[:, non_none_mask], axis=1)
    best_non_none_prob = y_prob[:, non_none_mask][np.arange(len(y_prob)), best_non_none_idx]
    none_prob = y_prob[:, none_idx]
    score = best_non_none_prob - none_prob
    
    is_event_true = (y_true != none_idx)
    
    def event_vs_none_metrics(t: float):
        pred_is_event = (score >= t)
        tp = np.sum(pred_is_event & is_event_true)
        fp = np.sum(pred_is_event & ~is_event_true)
        fn = np.sum(~pred_is_event & is_event_true)
        tn = np.sum(~pred_is_event & ~is_event_true)
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return tpr, fpr
    
    ts = np.linspace(-1.0, 1.0, 201)
    roc = []
    best_t, best_tpr = -1.0, 0.0
    for t in ts:
        tpr, fpr = event_vs_none_metrics(t)
        roc.append((float(t), float(tpr), float(fpr)))
        if fpr <= target_fpr and tpr > best_tpr:
            best_t, best_tpr = t, tpr
    
    return float(best_t), roc


def predict_with_none_bias(model: keras.Model, 
                           x: np.ndarray, 
                           none_idx: int, 
                           threshold: float) -> np.ndarray:
    """Apply biased decision rule at inference time.
    
    Predicts 'None' unless (best_non_none_prob - none_prob) >= threshold.
    
    Args:
        model: Trained classifier (softmax output)
        x: Input batch of embeddings
        none_idx: Index of None/background class
        threshold: Decision threshold from compute_none_bias_threshold
    
    Returns:
        pred: Predicted class indices
    """
    probs = model(x, training=False).numpy()
    num_classes = probs.shape[1]
    
    non_none_mask = np.ones(num_classes, dtype=bool)
    non_none_mask[none_idx] = False
    
    best_non_none_idx = np.argmax(probs[:, non_none_mask], axis=1)
    best_non_none_prob = probs[:, non_none_mask][np.arange(len(probs)), best_non_none_idx]
    none_prob = probs[:, none_idx]
    score = best_non_none_prob - none_prob
    
    global_non_none_indices = np.arange(num_classes)[non_none_mask]
    pred = np.full((len(probs),), fill_value=none_idx, dtype=int)
    choose_event = (score >= threshold)
    pred[choose_event] = global_non_none_indices[best_non_none_idx[choose_event]]
    return pred
