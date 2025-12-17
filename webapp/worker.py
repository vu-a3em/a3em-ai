import os
import numpy as np
import tensorflow as tf
import subprocess
from microesc import keras
from microesc.datasets.AugmentedDirectoryDataSet import AugmentedDirectoryDataSet
from microesc.classification.Yamnet import YamnetParams
from microesc.classification.yamnetmini import build_mini_yamnet_model
from sklearn.utils.class_weight import compute_class_weight
import shutil
import uuid
from collections import defaultdict

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
            uniform_classes_per_batch=True,
            background_classes=[], # Will be updated dynamically via uploads
            use_metadata=True
        )
        
    # Don't automatically inject background classes here - the UI decides via 'none_cap'
    global_state.log("Dataset initialized.")

    if global_state.base_model is None:
        params = config.params
        # Blocks definition matching the one used in tests
        blocks = [
            (64, [3, 3], 1),
            (128, [3, 3], 2),
            (128, [3, 3], 1),
            (256, [3, 3], 2),
            (256, [3, 3], 1),
        ]
        
        # Build full model
        full_model = build_mini_yamnet_model(params, blocks=blocks, dense_units=[])
        
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

def _convert_to_wav(src_path, target_sr):
    """Convert an audio file to WAV at the requested sample rate using ffmpeg.
    Returns the path to the created WAV file.
    Raises RuntimeError on failure.
    """
    dest = os.path.splitext(src_path)[0] + ".wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        src_path,
        "-ar",
        str(int(target_sr)),
        dest,
    ]
    try:
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Check if the output file was created
        if not os.path.exists(dest):
            raise RuntimeError(f"ffmpeg conversion failed: output file {dest} not created")
        
        # Remove the original file if it was not a WAV
        if os.path.splitext(src_path)[1].lower() != ".wav":
            os.remove(src_path)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found on system; required for audio conversion")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors='ignore') if e.stderr is not None else str(e)
        raise RuntimeError(f"ffmpeg conversion failed: {stderr}")
    return dest

def generate_embedding_dataset(seq, model):
    """Takes a KerasDataSet and returns (X, y) numpy arrays of embeddings."""
    embs = []
    lbls = []
    for i in range(len(seq)):
        batch_waveforms, batch_labels = seq[i]
        batch_embs = model(batch_waveforms, training=False)
        if hasattr(batch_embs, 'numpy'):
            batch_embs = batch_embs.numpy()
        embs.append(batch_embs)
        lbls.append(batch_labels)
    if len(embs) == 0:
        return np.array([]), np.array([])
    X = np.concatenate(embs, axis=0)
    y = np.concatenate(lbls, axis=0)
    return X, y

def remove_label_from_dataset(dataset, label_to_remove: str):
    """Remove all clips and references to a given label from a DirectoryDataSet/AugmentedDirectoryDataSet.
    Rebuild label_to_idx, idx_to_label, and label_counts.
    Returns True if the label was removed, False otherwise.
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

def add_file(file_obj, label, is_background, metadata_text):
    init_system()

    def _add_single(target_file):
        if target_file is None:
            raise ValueError("No file provided.")

        ext = os.path.splitext(target_file.name)[1].lower()
        if ext not in (".wav", ".flac", ".ogg", ".mp3", ".m4a"):
            raise ValueError("Unsupported file type.")

        label_dir = os.path.join(config.UPLOAD_ROOT, label if not is_background else "__background__")
        os.makedirs(label_dir, exist_ok=True)
        
        filename = f"{uuid.uuid4().hex}{ext}"
        dest = os.path.join(label_dir, filename)
        shutil.copy(target_file.name, dest)

        # Convert to WAV if necessary (use ffmpeg for broad container/codec support)
        if ext != ".wav":
            global_state.log(f"Converting {dest} to WAV format using ffmpeg.")
            try:
                dest_wav = _convert_to_wav(dest, config.TARGET_SAMPLE_RATE)
            except Exception as e:
                raise ValueError(f"Audio conversion failed: {e}")
            if not os.path.exists(dest_wav):
                raise ValueError(f"Conversion claimed success but {dest_wav} not found")
            dest = dest_wav
            global_state.log(f"Converted to WAV: {dest}")

        if metadata_text:
            meta_path = os.path.splitext(dest)[0] + ".meta"
            with open(meta_path, "w") as f:
                f.write(metadata_text)
                
        target_label = label if not is_background else "None"
        global_state.dataset.add_class_from_directory(
            label=target_label,
            path=label_dir,
            target_sample_rate_hz=config.TARGET_SAMPLE_RATE,
            target_clip_length=config.TARGET_CLIP_LENGTH,
            use_metadata=True,
            resplit=True
        )

        base_name = os.path.basename(target_file.name)
        return f"{base_name}: Added file to label '{target_label}'"

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
            per_bg_class = max(1, int(none_cap) // max(1, len(background_classes)))

            for bg_class in background_classes:
                bg_path = os.path.join(dataset_path, bg_class)
                if os.path.exists(bg_path):
                    global_state.dataset.add_class_from_directory(
                        label='None',
                        path=bg_path,
                        target_sample_rate_hz=config.TARGET_SAMPLE_RATE,
                        target_clip_length=config.TARGET_CLIP_LENGTH,
                        use_metadata=True,
                        max_samples=per_bg_class,
                        resplit=False
                    )
                    global_state.log(f"Added background class {bg_class} as 'None' (max_samples={per_bg_class})")
                else:
                    global_state.log(f"Warning: Background class {bg_class} missing at {bg_path}")

        # Perform the train/test split after any background additions
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
                none_indices = np.where(y_train == none_idx_check)[0]
                keep_none = np.random.choice(none_indices, size=none_cap, replace=False)
                non_none_indices = np.where(y_train != none_idx_check)[0]
                keep_indices = np.concatenate([non_none_indices, keep_none])
                # Sort indices to keep order (optional)
                keep_indices.sort()
                X_train = X_train[keep_indices]
                y_train = y_train[keep_indices]

        global_state.log(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples.")

        # Build classifier head
        input_shape = base_model.output.shape[1:]
        
        hidden_units = train_params.get('hidden_units', [256, 128])
        activation = train_params.get('activation', 'gelu')
        dropout = train_params.get('dropout', 0.25)
        
        layers = [keras.layers.Input(shape=input_shape, dtype='float32')]
        for units in hidden_units:
            layers.append(keras.layers.Dense(units=units, activation=activation))
            layers.append(keras.layers.Dropout(dropout))
        
        # Output layer
        # dataset.label_to_idx contains all classes including "None" if added
        num_classes = len(dataset.label_to_idx)
        layers.append(keras.layers.Dense(units=num_classes, activation='softmax'))
        
        classifier = keras.Sequential(layers)
        
        # Compile
        lr = train_params.get('lr', config.DEFAULT_LEARNING_RATE)
        classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                           loss=keras.losses.SparseCategoricalCrossentropy(),
                           metrics=[keras.metrics.SparseCategoricalAccuracy()])
        
        # Class weights
        classes = np.unique(y_train)
        cw = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}
        
        # Callbacks
        patience = train_params.get('patience', config.DEFAULT_PATIENCE)
        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        
        class LogCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                msg = f"Epoch {epoch+1}: loss={logs['loss']:.4f}, val_loss={logs['val_loss']:.4f}, acc={logs['sparse_categorical_accuracy']:.4f}"
                global_state.log(msg)
        
        epochs = train_params.get('epochs', config.DEFAULT_EPOCHS)
        classifier.fit(X_train, y_train, 
                       validation_data=(X_test, y_test) if len(X_test) > 0 else None,
                       epochs=epochs,
                       callbacks=[callback, LogCallback()],
                       class_weight=class_weight_dict,
                       verbose=0)
        
        global_state.classifier = classifier
        global_state.log("Training complete.")
        
        # Compute None bias threshold
        none_label = "None"
        if none_label in dataset.label_to_idx and len(X_test) > 0:
            global_state.log("Computing None-bias threshold...")
            none_idx = dataset.label_to_idx[none_label]
            global_state.none_idx = none_idx
            
            probs = classifier.predict(X_test)
            y_true = y_test
            
            num_classes = probs.shape[1]
            non_none_mask = np.ones(num_classes, dtype=bool)
            non_none_mask[none_idx] = False
            
            best_non_none_idx = np.argmax(probs[:, non_none_mask], axis=1)
            best_non_none_prob = probs[:, non_none_mask][np.arange(len(probs)), best_non_none_idx]
            none_prob = probs[:, none_idx]
            score = best_non_none_prob - none_prob
            
            is_event_true = (y_true != none_idx)
            
            ts = np.linspace(-1.0, 1.0, 201)
            roc = []
            best_t, best_tpr = ts[0], -1.0
            target_fpr = train_params.get('target_fpr', 0.10)
            
            for t in ts:
                choose_event = (score >= t)
                # pred logic
                # We only care about TPR/FPR here
                is_event_pred = choose_event
                
                tp = np.sum(is_event_pred & is_event_true)
                fp = np.sum(is_event_pred & ~is_event_true)
                fn = np.sum(~is_event_pred & is_event_true)
                tn = np.sum(~is_event_pred & ~is_event_true)
                
                tpr = tp / (tp + fn + 1e-9)
                fpr = fp / (fp + tn + 1e-9)
                
                roc.append((float(t), float(tpr), float(fpr)))
                if fpr <= target_fpr and tpr > best_tpr:
                    best_tpr = tpr
                    best_t = t
            
            global_state.none_bias_threshold = float(best_t)
            global_state.roc_curve = roc
            global_state.log(f"Selected None-bias threshold={best_t:.4f} (TPR={best_tpr:.4f}, FPR<={target_fpr})")
            
        global_state.set_status("Done")

    except Exception as e:
        global_state.log(f"Error during training: {str(e)}")
        global_state.set_status("Error")
        # raise e # Don't raise in thread, just log

def predict(audio_file, threshold_override=None):
    init_system()
    if global_state.classifier is None:
        return "No model trained.", 0.0, None
    
    if audio_file is None:
        return "No audio file.", 0.0, None

    # Preprocess
    # We need to load audio, resample, split into clips?
    # For simplicity, let's assume short clips or take the first clip.
    # We can use AugmentedDirectoryDataSet's _process_audio_file logic or similar.
    # But that's internal.
    # Let's use librosa directly or reuse AudioClip logic.
    
    # We need to extract embeddings.
    # We can create a temporary AudioClip and use the base model.
    
    # Save temp file
    tmp_path = f"/tmp/{uuid.uuid4().hex}.wav"
    shutil.copy(audio_file.name, tmp_path)
    
    try:
        # Use base model to get embeddings
        # Yamnet expects waveform
        import librosa
        wav, sr = librosa.load(tmp_path, sr=config.TARGET_SAMPLE_RATE)
        
        # Pad or trim
        target_len = int(config.TARGET_CLIP_LENGTH * config.TARGET_SAMPLE_RATE)
        if len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)))
        else:
            wav = wav[:target_len]
            
        wav = wav.astype(np.float32)
        # Normalize? Yamnet usually expects -1 to 1 or similar.
        if np.max(np.abs(wav)) > 0:
            wav = wav / np.max(np.abs(wav))
            
        # Add batch dim
        wav_batch = np.expand_dims(wav, axis=0)
        
        embeddings = global_state.base_model(wav_batch, training=False).numpy()
        
        probs = global_state.classifier.predict(embeddings)
        
        # Apply threshold
        threshold = threshold_override if threshold_override is not None else global_state.none_bias_threshold
        none_idx = global_state.none_idx
        
        if none_idx is not None:
            num_classes = probs.shape[1]
            non_none_mask = np.ones(num_classes, dtype=bool)
            non_none_mask[none_idx] = False
            
            best_non_none_idx = np.argmax(probs[:, non_none_mask], axis=1)[0]
            best_non_none_prob = probs[:, non_none_mask][0, best_non_none_idx]
            none_prob = probs[0, none_idx]
            score = best_non_none_prob - none_prob
            
            if score >= threshold:
                # Map back to global index
                global_indices = np.arange(num_classes)[non_none_mask]
                pred_idx = global_indices[best_non_none_idx]
            else:
                pred_idx = none_idx
        else:
            pred_idx = np.argmax(probs[0])
            
        label = global_state.dataset.idx_to_label[pred_idx]
        confidence = float(np.max(probs[0]))
        
        return label, confidence, None # Plot?

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
