"""Dataset manipulation utilities for few-shot learning."""

import os
import glob
import random
from typing import List
from collections import defaultdict
import numpy as np


def remove_label_from_dataset(dataset, label_to_remove: str) -> bool:
    """Remove all clips and references to a given label from a DirectoryDataSet/AugmentedDirectoryDataSet.
    
    Rebuilds label_to_idx, idx_to_label, and label_counts after removal.
    
    Args:
        dataset: DirectoryDataSet or AugmentedDirectoryDataSet instance
        label_to_remove: Label string to remove
    
    Returns:
        True if the label was removed, False if it didn't exist
    """
    if label_to_remove not in dataset.label_to_idx:
        return False

    # Remove clips for the label
    dataset.clips = [c for c in dataset.clips if c.label != label_to_remove]

    # Remove label from set and counts
    try:
        dataset.labels.discard(label_to_remove)
    except Exception:
        pass
    if label_to_remove in dataset.label_counts:
        del dataset.label_counts[label_to_remove]

    # Rebuild label_to_idx & idx_to_label from remaining labels
    remaining_labels = sorted(list(set(c.label for c in dataset.clips)))
    dataset.label_to_idx = {lbl: idx for idx, lbl in enumerate(remaining_labels)}
    dataset.idx_to_label = {idx: lbl for lbl, idx in dataset.label_to_idx.items()}

    # Update clip label idx values and recompute label_counts
    dataset.label_counts = defaultdict(int)
    for c in dataset.clips:
        c.label_idx = dataset.label_to_idx[c.label]
        dataset.label_counts[c.label] += 1

    # Resplit train/test
    try:
        dataset._split_train_test()
    except Exception:
        pass

    return True


def add_background_as_none(dataset,
                           dataset_path: str,
                           background_classes: List[str],
                           max_none_samples: int,
                           target_sample_rate_hz: int,
                           target_clip_length: float,
                           use_metadata: bool = True) -> int:
    """Add background noise classes as 'None' class to a dataset.
    
    This is used for few-shot learning to teach the model to reject non-target sounds.
    
    Args:
        dataset: AugmentedDirectoryDataSet instance
        dataset_path: Path to directory containing background class folders
        background_classes: List of background class folder names (e.g., ['Noise', 'Wind'])
        max_none_samples: Maximum total number of 'None' samples to add
        target_sample_rate_hz: Target sample rate for audio
        target_clip_length: Target clip length in seconds
        use_metadata: Whether to use metadata files for clip boundaries
    
    Returns:
        Number of background samples added
    """
    per_bg_class = max(1, max_none_samples // max(1, len(background_classes)))
    total_added = 0
    
    for bg_class in background_classes:
        bg_path = os.path.join(dataset_path, bg_class)
        if os.path.exists(bg_path):
            try:
                dataset.add_class_from_directory(
                    label='None',
                    path=bg_path,
                    target_sample_rate_hz=target_sample_rate_hz,
                    target_clip_length=target_clip_length,
                    max_samples=per_bg_class,
                    use_metadata=use_metadata,
                    resplit=False  # Will resplit once at the end
                )
                total_added += min(per_bg_class, len(glob.glob(os.path.join(bg_path, '*.*'))))
            except Exception as e:
                print(f"Warning: Failed to add background class {bg_class}: {e}")
    
    # Resplit after all additions
    try:
        dataset._split_train_test()
    except Exception:
        pass
    
    return total_added


def downsample_none_class(X_train: np.ndarray, 
                         y_train: np.ndarray,
                         none_idx: int,
                         max_none_samples: int) -> tuple:
    """Downsample the 'None' class in training data to prevent dominance.
    
    Args:
        X_train: Training features
        y_train: Training labels
        none_idx: Index of 'None' class
        max_none_samples: Maximum number of 'None' samples to keep
    
    Returns:
        (X_train_downsampled, y_train_downsampled)
    """
    none_indices = np.where(y_train == none_idx)[0]
    none_count = len(none_indices)
    
    if none_count <= max_none_samples:
        return X_train, y_train
    
    # Randomly select subset of None samples
    keep_none = np.random.choice(none_indices, size=max_none_samples, replace=False)
    non_none_indices = np.where(y_train != none_idx)[0]
    keep_indices = np.concatenate([non_none_indices, keep_none])
    
    # Sort to maintain some order
    keep_indices.sort()
    
    return X_train[keep_indices], y_train[keep_indices]
