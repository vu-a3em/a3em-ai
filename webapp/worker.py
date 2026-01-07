import os
import numpy as np
import tensorflow as tf
from microesc import keras
from microesc.datasets.AugmentedDirectoryDataSet import AugmentedDirectoryDataSet
from microesc.datasets.DirectoryDataSet import AudioClip
from microesc.detection.SpectralFluxDetector import SpectralFluxDetector
from microesc.classification.Yamnet import YamnetParams
from microesc.classification.yamnetmini import build_mini_yamnet_model, DEFAULT_BLOCKS
from microesc.classification.training import (
    TrainingConfig, generate_embedding_dataset, build_classifier_head,
    compute_none_bias_threshold, predict_with_none_bias
)
from microesc.datasets.utils import remove_label_from_dataset, add_background_as_none, downsample_none_class
from microesc.tools.audio import convert_to_wav
from sklearn.utils.class_weight import compute_class_weight
import shutil
import uuid
import librosa

from . import config
from .state import global_state

def init_system():
    """Initialize dataset and base model if not already done."""
    if global_state.dataset is None:
        os.makedirs(config.UPLOAD_ROOT, exist_ok=True)
        # Initialize dataset with upload root
        global_state.dataset = AugmentedDirectoryDataSet(
            base_path=config.UPLOAD_ROOT,
            target_sample_rate_hz=config.TARGET_SAMPLE_RATE,
            target_clip_length=config.TARGET_CLIP_LENGTH,
            training_split_percent=config.DEFAULT_TRAIN_SPLIT,
            uniform_classes_per_batch=False,  # defer balancing until training if requested
            background_classes=[], # Will be updated dynamically via uploads
            use_metadata=True
        )
        
    # Don't automatically inject background classes here - the UI decides via 'none_cap'
    global_state.log("Dataset initialized.")

    if global_state.base_model is None:
        params = config.params
        
        # Build full model using default architecture
        full_model = build_mini_yamnet_model(params, blocks=DEFAULT_BLOCKS, dense_units=[])
        
        # Load weights if available
        model_path = os.path.join(config.BASE_DIR, "yamnetmini--1.keras")
        if os.path.exists(model_path):
             try:
                full_model.load_weights(model_path)
                global_state.log(f"Loaded base model weights from {model_path}")
             except Exception as e:
                global_state.log(f"Failed to load weights: {e}")
        else:
            global_state.log("No pretrained weights found. Using random initialization (not recommended for few-shot).")
        
        # Remove the classification head to get embeddings
        # build_mini_yamnet_model adds a Dense layer at the end.
        full_model.pop() 
        global_state.base_model = full_model
        global_state.log("Base model (feature extractor) ready.")


def add_file(file_obj, label, is_background, metadata_text):
    init_system()

    def _add_single(target_file):
        if target_file is None:
            raise ValueError("No file provided.")

        dest = None

        ext = os.path.splitext(target_file.name)[1].lower()
        if ext not in (".wav", ".flac", ".ogg", ".mp3", ".m4a"):
            raise ValueError("Unsupported file type.")

        try:
            label_dir = os.path.join(config.UPLOAD_ROOT, label if not is_background else "__background__")
            os.makedirs(label_dir, exist_ok=True)

            filename = f"{uuid.uuid4().hex}{ext}"
            dest = os.path.join(label_dir, filename)
            shutil.copy(target_file.name, dest)

            # Convert to WAV if necessary (use ffmpeg for broad container/codec support)
            if ext != ".wav":
                global_state.log(f"Converting {dest} to WAV format using ffmpeg.")
                try:
                    dest = convert_to_wav(dest, config.TARGET_SAMPLE_RATE, remove_original=True)
                    global_state.log(f"Converted to WAV: {dest}")
                except Exception as e:
                    raise ValueError(f"Audio conversion failed: {e}")

            # Check if already processed
            if dest in global_state.processed_files:
                base_name = os.path.basename(target_file.name)
                return f"{base_name}: Already processed (skipped)"

            target_label = label if not is_background else "None"
            meta_path = os.path.splitext(dest)[0] + ".meta"

            # Handle metadata: either provided by user or auto-generate with event detector
            if metadata_text:
                # User provided metadata
                with open(meta_path, "w") as f:
                    f.write(metadata_text)
                use_metadata = True
            elif global_state.detector_config.get('enabled', False) and global_state.detector_config.get('type') == 'spectral_flux':
                # Auto-generate metadata using event detector
                global_state.log(f"Running event detection on {os.path.basename(dest)}...")
                detector = SpectralFluxDetector(
                    global_state.detector_config.get('threshold', config.DEFAULT_DETECTOR_THRESHOLD),
                    config.TARGET_SAMPLE_RATE,
                    config.DEFAULT_FFT_LENGTH,
                    config.DEFAULT_HOP_LENGTH,
                    False,
                    config.DEFAULT_MIN_FREQ,
                    config.DEFAULT_MAX_FREQ,
                    global_state.detector_config.get('min_gap', config.DEFAULT_DETECTOR_MIN_GAP)
                )

                try:
                    event_times = detector.detect_events(dest)

                    # Write metadata file
                    with open(meta_path, "w") as f:
                        for t in event_times:
                            f.write(f"{t:.6f},{target_label}\n")

                    global_state.log(f"Detected {len(event_times)} events in {os.path.basename(dest)}")
                    use_metadata = True
                except Exception as e:
                    # Log and abort processing this file — do not continue with full-file fallback
                    global_state.log(f"Event detection failed: {e}. Aborting file addition.")
                    try:
                        if dest and os.path.exists(dest):
                            os.remove(dest)
                    except Exception:
                        pass
                    raise RuntimeError(f"Event detection failed for {os.path.basename(dest)}: {e}")
            else:
                # No metadata, no event detection - use full file
                use_metadata = False

            # Add to dataset using add_clips_for_label for efficiency (avoids directory rescan)
            # We'll manually create clips and add them
            if use_metadata and os.path.exists(meta_path):
                # Load metadata and create clips
                clips_to_add = []
                with open(meta_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.rsplit(',', 1)
                        if len(parts) == 2:
                            try:
                                start_time = float(parts[0])
                                clip_label = parts[1].strip()
                                if clip_label.lower() in ('ignore', 'unknown'):
                                    continue

                                # Ensure label exists in dataset
                                if clip_label not in global_state.dataset.label_to_idx:
                                    idx = len(global_state.dataset.label_to_idx)
                                    global_state.dataset.label_to_idx[clip_label] = idx
                                    global_state.dataset.idx_to_label[idx] = clip_label
                                    global_state.dataset.label_counts[clip_label] = 0

                                label_idx = global_state.dataset.label_to_idx[clip_label]
                                end_time = start_time + config.TARGET_CLIP_LENGTH

                                clip = AudioClip(
                                    label_idx=label_idx,
                                    label=clip_label,
                                    path=dest,
                                    start_seconds=start_time,
                                    end_seconds=end_time,
                                    target_sample_rate_hz=config.TARGET_SAMPLE_RATE
                                )
                                clips_to_add.append(clip)
                            except ValueError:
                                continue

                if clips_to_add:
                    # Group clips by their metadata label so we add them under the correct label
                    grouped = {}
                    for c in clips_to_add:
                        grouped.setdefault(c.label, []).append(c)
                    num_clips = 0
                    # Log pre-add counts
                    pre_counts = {lbl: global_state.dataset.label_counts.get(lbl, 0) for lbl in grouped.keys()}
                    global_state.log(f"Pre-add label counts: {pre_counts}")
                    for lbl, group in grouped.items():
                        # Ensure label exists in dataset mapping (add_clips_for_label will also ensure this)
                        # Temporarily disable uniform augmentation during incremental upload to avoid altering other labels
                        old_uniform = getattr(global_state.dataset, 'uniform_classes_per_batch', False)
                        try:
                            global_state.dataset.uniform_classes_per_batch = False
                            global_state.dataset.add_clips_for_label(lbl, group, is_background=is_background, resplit=False)
                        finally:
                            global_state.dataset.uniform_classes_per_batch = old_uniform
                        added = len(group)
                        num_clips += added
                        post = global_state.dataset.label_counts.get(lbl, 0)
                        global_state.log(f"Added {added} clip(s) to label '{lbl}' (now {post})")
            else:
                # No metadata - add entire file as single clip
                if target_label not in global_state.dataset.label_to_idx:
                    idx = len(global_state.dataset.label_to_idx)
                    global_state.dataset.label_to_idx[target_label] = idx
                    global_state.dataset.idx_to_label[idx] = target_label
                    global_state.dataset.label_counts[target_label] = 0

                label_idx = global_state.dataset.label_to_idx[target_label]
                clip = AudioClip(
                    label_idx=label_idx,
                    label=target_label,
                    path=dest,
                    start_seconds=0.0,
                    end_seconds=config.TARGET_CLIP_LENGTH,
                    target_sample_rate_hz=config.TARGET_SAMPLE_RATE
                )
                # Log pre-add count for single clip
                pre_single = global_state.dataset.label_counts.get(target_label, 0)
                global_state.log(f"Pre-add single clip count for '{target_label}': {pre_single}")
                # Temporarily disable uniform augmentation during incremental upload
                old_uniform = getattr(global_state.dataset, 'uniform_classes_per_batch', False)
                try:
                    global_state.dataset.uniform_classes_per_batch = False
                    global_state.dataset.add_clips_for_label(target_label, [clip], is_background=is_background, resplit=False)
                finally:
                    global_state.dataset.uniform_classes_per_batch = old_uniform
                post_single = global_state.dataset.label_counts.get(target_label, 0)
                global_state.log(f"Added 1 clip to '{target_label}' (now {post_single})")
                num_clips = 1

            # Mark as processed
            if dest:
                global_state.processed_files.add(dest)

            base_name = os.path.basename(target_file.name)
            return f"{base_name}: Added {num_clips} clip(s) to label '{target_label}'"
        except Exception:
            # Cleanup partial file if something went wrong
            try:
                if dest and os.path.exists(dest):
                    os.remove(dest)
            except Exception:
                pass
            raise

    if isinstance(file_obj, (list, tuple)):
        messages = []
        for item in file_obj:
            try:
                messages.append(_add_single(item))
            except Exception as exc:
                print(f"Error adding file: {exc}")
                file_name = getattr(item, "name", "<unknown>")
                messages.append(f"{file_name}: {str(exc)}")
        return "\n".join(messages)

    try:
        return _add_single(file_obj)
    except Exception as exc:
        print(f"Error adding file: {exc}")
        return str(exc)

def train_model(train_params):
    init_system()
    global_state.set_status("Training")
    global_state.log("Starting training...")
    
    try:
        dataset = global_state.dataset
        base_model = global_state.base_model
        
        # Update dataset params
        dataset.training_split_percent = train_params.get('split', config.DEFAULT_TRAIN_SPLIT)

        # Optionally add background (None) classes based on 'none_cap' training param.
        none_cap = train_params.get('none_cap', None)
        if none_cap is not None and none_cap > 0:
            # Remove existing 'None' if present to allow re-adding with new cap
            if 'None' in dataset.label_to_idx:
                removed = remove_label_from_dataset(dataset, 'None')
                global_state.log(f"Removed existing 'None' class: {removed}")

            background_classes = ['Noise', 'VehicleExhaust', 'Wind']
            dataset_path = '/isis/home/steing/AIDataSet'
            
            num_added = add_background_as_none(
                dataset=dataset,
                dataset_path=dataset_path,
                background_classes=background_classes,
                max_none_samples=int(none_cap),
                target_sample_rate_hz=config.TARGET_SAMPLE_RATE,
                target_clip_length=config.TARGET_CLIP_LENGTH,
                use_metadata=True
            )
            global_state.log(f"Added {num_added} background samples as 'None' class")

        # Set balancing behavior from training params and perform augmentation if requested
        balance = train_params.get('balance', False)
        dataset.uniform_classes_per_batch = bool(balance)
        if dataset.uniform_classes_per_batch:
            try:
                global_state.log("Applying uniform class augmentation before split...")
                dataset._augment_uniform_classes()
            except Exception as e:
                global_state.log(f"Uniform augmentation failed: {e}")

        global_state.log("Splitting dataset into train/test...")
        dataset._split_train_test()
        
        batch_size = train_params.get('batch_size', config.DEFAULT_BATCH_SIZE)
        train_ds = dataset.train_dataset(batch_size=batch_size)
        test_ds = dataset.test_dataset(batch_size=batch_size)
        
        if len(dataset.clips) == 0:
            raise ValueError("Dataset is empty.")

        global_state.log("Extracting embeddings...")
        X_train, y_train = generate_embedding_dataset(train_ds, base_model)
        X_test, y_test = generate_embedding_dataset(test_ds, base_model)
        
        if len(X_train) == 0:
             raise ValueError("No training data extracted.")

        # Debug logging: show label mapping and counts
        try:
            global_state.log(f"Dataset label_to_idx: {dataset.label_to_idx}")
            for idx, label in dataset.idx_to_label.items():
                count = dataset.label_counts.get(label, 0)
                global_state.log(f"  idx {idx} => '{label}' : {count} samples")
        except Exception:
            global_state.log("Error while logging dataset label info")

        # Show distribution of y_train mapped to label names
        try:
            u, c = np.unique(y_train, return_counts=True)
            dist = {dataset.idx_to_label[int(ui)]: int(ci) for ui, ci in zip(u, c)}
            global_state.log(f"y_train distribution: {dist}")
        except Exception:
            global_state.log(f"y_train values: {set(y_train.tolist()) if hasattr(y_train, 'tolist') else 'unknown'}")

        # Basic checks: need at least two classes
        if len(np.unique(y_train)) < 2:
            global_state.log("Insufficient class diversity for training (only one class present). Aborting training.")
            global_state.set_status("Error: insufficient class diversity")
            return

        # If 'None' class present, log details
        if 'None' in dataset.label_to_idx:
            none_idx_check = dataset.label_to_idx['None']
            none_train = int(np.sum(y_train == none_idx_check))
            none_val = int(np.sum(y_test == none_idx_check)) if len(y_test) > 0 else 0
            global_state.log(f"'None' class idx={none_idx_check} train_samples={none_train}, val_samples={none_val}")
            if none_train >= len(y_train):
                global_state.log("All training samples belong to 'None' class. Aborting training.")
                global_state.set_status("Error: only None class in training")
                return
            # Optional downsample of 'None' class to limit its dominance
            none_cap = train_params.get('none_cap', None)
            if none_cap is not None and none_train > none_cap:
                global_state.log(f"Downsampling 'None' class from {none_train} to {none_cap} samples for training.")
                X_train, y_train = downsample_none_class(X_train, y_train, none_idx_check, none_cap)

        global_state.log(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples.")

        # Build classifier head using shared utility
        input_shape = base_model.output.shape[1:]
        num_classes = len(dataset.label_to_idx)
        
        training_config = TrainingConfig(
            batch_size=train_params.get('batch_size', config.DEFAULT_BATCH_SIZE),
            learning_rate=train_params.get('lr', config.DEFAULT_LEARNING_RATE),
            epochs=train_params.get('epochs', config.DEFAULT_EPOCHS),
            patience=train_params.get('patience', config.DEFAULT_PATIENCE),
            hidden_units=train_params.get('hidden_units', [256, 128]),
            activation=train_params.get('activation', 'relu'),
            dropout=train_params.get('dropout', 0.25),
            target_fpr=train_params.get('target_fpr', 0.10)
        )
        
        classifier = build_classifier_head(input_shape, num_classes, training_config)
        
        # Compile
        classifier.compile(
            optimizer=keras.optimizers.Adam(learning_rate=training_config.learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy()]
        )
        
        # Class weights
        classes = np.unique(y_train)
        cw = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}
        
        # Callbacks
        callback = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=training_config.patience, 
            restore_best_weights=True
        )
        
        class LogCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                msg = f"Epoch {epoch+1}: loss={logs['loss']:.4f}, val_loss={logs['val_loss']:.4f}, acc={logs['sparse_categorical_accuracy']:.4f}"
                global_state.log(msg)
        
        classifier.fit(X_train, y_train, 
                       validation_data=(X_test, y_test) if len(X_test) > 0 else None,
                       epochs=training_config.epochs,
                       callbacks=[callback, LogCallback()],
                       class_weight=class_weight_dict,
                       verbose=0)
        
        global_state.classifier = classifier
        global_state.log("Training complete.")
        
        # Compute None bias threshold using shared utility
        none_label = "None"
        if none_label in dataset.label_to_idx and len(X_test) > 0:
            global_state.log("Computing None-bias threshold...")
            none_idx = dataset.label_to_idx[none_label]
            global_state.none_idx = none_idx
            
            # Create a simple tf.data.Dataset for the shared function
            test_ds_for_threshold = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
            
            best_t, roc = compute_none_bias_threshold(
                classifier, 
                test_ds_for_threshold, 
                none_idx, 
                training_config.target_fpr
            )
            
            global_state.none_bias_threshold = float(best_t)
            global_state.roc_curve = roc
            
            # Find TPR for logging
            best_tpr = next((tpr for t, tpr, fpr in roc if abs(t - best_t) < 0.001), 0.0)
            global_state.log(f"Selected None-bias threshold={best_t:.4f} (TPR={best_tpr:.4f}, FPR<={training_config.target_fpr})")
            
        global_state.set_status("Done")

    except Exception as e:
        global_state.log(f"Error during training: {str(e)}")
        global_state.set_status("Error")
        # raise e # Don't raise in thread, just log

def predict(audio_file, threshold_override=None, metadata_text=None):
    """
    Perform event-level predictions on an audio file.
    
    Args:
        audio_file: File object containing audio
        threshold_override: Optional manual None-bias threshold
        metadata_text: Optional manual event timestamps (format: "timestamp,label\n")
    
    Returns:
        List of (timestamp, label, confidence) tuples for each detected event
    """
    init_system()
    if global_state.classifier is None:
        return [("No model trained", 0.0, 0.0)]
    
    if audio_file is None:
        return [("No audio file", 0.0, 0.0)]

    # Save temp file
    tmp_path = f"/tmp/{uuid.uuid4().hex}.wav"
    shutil.copy(audio_file.name, tmp_path)
    
    try:
        # Load full audio
        wav, sr = librosa.load(tmp_path, sr=config.TARGET_SAMPLE_RATE)
        
        # Cache for visualization
        global_state.last_audio_data = (wav.copy(), sr)
        
        # Get event timestamps
        event_times = []
        
        if metadata_text:
            # Use provided metadata timestamps
            for line in metadata_text.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 1:
                    try:
                        timestamp = float(parts[0])
                        event_times.append(timestamp)
                    except ValueError:
                        continue
        elif global_state.detector_config.get('enabled', True):
            # Use event detector
            detector = SpectralFluxDetector(
                global_state.detector_config.get('threshold', config.DEFAULT_DETECTOR_THRESHOLD),
                config.TARGET_SAMPLE_RATE,
                config.DEFAULT_FFT_LENGTH,
                config.DEFAULT_HOP_LENGTH,
                False,
                config.DEFAULT_MIN_FREQ,
                config.DEFAULT_MAX_FREQ,
                global_state.detector_config.get('min_gap', config.DEFAULT_DETECTOR_MIN_GAP)
            )
            event_times = detector.detect_events(tmp_path)
        else:
            # Fallback: sliding window
            hop_size = config.TARGET_CLIP_LENGTH / 2  # 50% overlap
            audio_duration = len(wav) / sr
            event_times = list(np.arange(0, audio_duration, hop_size))
        
        if len(event_times) == 0:
            return [("No events detected", 0.0, 0.0)]
        
        # Extract segments and get embeddings for each event
        target_len = int(config.TARGET_CLIP_LENGTH * config.TARGET_SAMPLE_RATE)
        all_embeddings = []
        valid_times = []
        
        for t in event_times:
            start_sample = int(t * sr)
            end_sample = start_sample + target_len
            
            # Extract segment
            if start_sample >= len(wav):
                continue
                
            segment = wav[start_sample:end_sample]
            
            # Pad if needed
            if len(segment) < target_len:
                segment = np.pad(segment, (0, target_len - len(segment)))
            
            segment = segment.astype(np.float32)
            
            # Normalize
            if np.max(np.abs(segment)) > 0:
                segment = segment / np.max(np.abs(segment))
            
            # Get embeddings
            segment_batch = np.expand_dims(segment, axis=0)
            embeddings = global_state.base_model(segment_batch, training=False).numpy()
            all_embeddings.append(embeddings[0])
            valid_times.append(t)
        
        if len(all_embeddings) == 0:
            return [("No valid segments", 0.0, 0.0)]
        
        # Batch predict all segments
        embeddings_array = np.array(all_embeddings)
        probs = global_state.classifier.predict(embeddings_array, verbose=0)
        
        # Apply None-bias threshold
        threshold = threshold_override if threshold_override is not None else global_state.none_bias_threshold
        none_idx = global_state.none_idx
        
        results = []
        for i, (t, prob_vec) in enumerate(zip(valid_times, probs)):
            if none_idx is not None and threshold is not None:
                # Apply bias correction
                pred_idx = predict_with_none_bias(
                    global_state.classifier,
                    embeddings_array[i:i+1],
                    none_idx,
                    threshold
                )[0]
            else:
                pred_idx = np.argmax(prob_vec)
            
            label = global_state.dataset.idx_to_label[pred_idx]
            confidence = float(prob_vec[pred_idx])
            results.append((t, label, confidence))
        
        # Cache results and embeddings for threshold adjustment
        global_state.last_inference_results = results
        global_state.last_inference_embeddings = embeddings_array
        
        return results

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def adjust_threshold(new_threshold):
    """
    Recalculate predictions using cached embeddings with a new threshold.
    
    Args:
        new_threshold: New None-bias threshold value
    
    Returns:
        List of (timestamp, label, confidence) tuples with updated predictions
    """
    if global_state.last_inference_embeddings is None:
        return [("No cached predictions", 0.0, 0.0)]
    
    if global_state.classifier is None:
        return [("No model trained", 0.0, 0.0)]
    
    # Get predictions from cached embeddings
    embeddings_array = global_state.last_inference_embeddings
    probs = global_state.classifier.predict(embeddings_array, verbose=0)
    
    none_idx = global_state.none_idx
    results = []
    
    # Extract original timestamps
    original_times = [r[0] for r in global_state.last_inference_results]
    
    for i, (t, prob_vec) in enumerate(zip(original_times, probs)):
        if none_idx is not None and new_threshold is not None:
            # Apply bias correction with new threshold
            pred_idx = predict_with_none_bias(
                global_state.classifier,
                embeddings_array[i:i+1],
                none_idx,
                new_threshold
            )[0]
        else:
            pred_idx = np.argmax(prob_vec)
        
        label = global_state.dataset.idx_to_label[pred_idx]
        confidence = float(prob_vec[pred_idx])
        results.append((t, label, confidence))
    
    # Update cached results
    global_state.last_inference_results = results
    
    return results


def evaluate_test_set(audio_files, metadata_files):
    """
    Evaluate model performance on a test set with ground truth labels.
    
    Args:
        audio_files: List of audio file objects
        metadata_files: List of metadata strings (CSV format: timestamp,label)
    
    Returns:
        Dictionary with confusion matrix, classification report, and metrics
    """
    init_system()
    
    if global_state.classifier is None:
        return {"error": "No model trained"}
    
    if not audio_files or not metadata_files:
        return {"error": "No test files provided"}
    
    from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
    
    all_true_labels = []
    all_pred_labels = []
    all_true_indices = []
    all_pred_indices = []
    
    # Process each test file
    for audio_file, metadata_text in zip(audio_files, metadata_files):
        if not metadata_text or not audio_file:
            continue
        
        # Parse metadata to get ground truth
        ground_truth = {}  # timestamp -> label
        for line in metadata_text.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    timestamp = float(parts[0])
                    label = parts[1].strip()
                    if label.lower() not in ('ignore', 'unknown'):
                        ground_truth[timestamp] = label
                except ValueError:
                    continue
        
        if not ground_truth:
            continue
        
        # Get predictions using the same timestamps
        timestamps_text = '\n'.join([f"{t}" for t in ground_truth.keys()])
        predictions = predict(audio_file, metadata_text=timestamps_text)
        
        # Match predictions to ground truth
        for pred_time, pred_label, _ in predictions:
            # Find closest ground truth timestamp (within a small window)
            closest_gt = min(ground_truth.keys(), key=lambda t: abs(t - pred_time), default=None)
            if closest_gt is not None and abs(closest_gt - pred_time) < 0.5:  # 0.5s tolerance
                true_label = ground_truth[closest_gt]
                all_true_labels.append(true_label)
                all_pred_labels.append(pred_label)
                
                # Convert to indices
                if true_label in global_state.dataset.label_to_idx:
                    all_true_indices.append(global_state.dataset.label_to_idx[true_label])
                    all_pred_indices.append(global_state.dataset.label_to_idx[pred_label])
    
    if len(all_true_labels) == 0:
        return {"error": "No matching predictions found"}
    
    # Compute metrics
    labels_list = sorted(set(all_true_labels + all_pred_labels))
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=labels_list)
    report = classification_report(all_true_labels, all_pred_labels, labels=labels_list, output_dict=True)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_true_labels, all_pred_labels, labels=labels_list, average=None
    )
    
    per_class = {}
    for i, label in enumerate(labels_list):
        per_class[label] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    # Store in global state
    global_state.evaluation_metrics = {
        'confusion_matrix': cm.tolist(),
        'labels': labels_list,
        'classification_report': report,
        'per_class_metrics': per_class
    }
    
    return global_state.evaluation_metrics


def save_trained_model():
    """
    Save the trained classifier model to a file.
    
    Returns:
        Path to saved model file, or error message
    """
    if global_state.classifier is None:
        return None, "No trained model available."
    
    try:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_path = f"/tmp/a3em_classifier_{timestamp}.keras"
        
        global_state.classifier.save(model_path)
        global_state.log(f"Model saved to {model_path}")
        
        return model_path, f"Model saved successfully"
    except Exception as e:
        error_msg = f"Error saving model: {str(e)}"
        global_state.log(error_msg)
        return None, error_msg
