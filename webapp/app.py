import gradio as gr
import threading
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from . import worker
from . import config
from .state import global_state

def get_dataset_summary():
    if global_state.dataset is None:
        return "Dataset not initialized."
    
    counts = global_state.dataset.label_counts
    df = pd.DataFrame(list(counts.items()), columns=["Label", "Count"])
    return df

def upload_file_handler(file_obj, label, is_background, metadata, det_enabled, det_type, det_threshold, det_min_gap):
    if not file_obj:
        return "No files selected.", get_dataset_summary()
    
    # Update global detector config
    global_state.detector_config['enabled'] = det_enabled
    global_state.detector_config['type'] = 'spectral_flux' if det_type == "Spectral Flux" else None
    global_state.detector_config['threshold'] = det_threshold
    global_state.detector_config['min_gap'] = det_min_gap

    msg = worker.add_file(file_obj, label, is_background, metadata)
    return msg, get_dataset_summary()

def start_training_handler(split, batch_size, epochs, lr, hidden_units_str, activation, dropout, target_fpr, none_cap, balance):
    if global_state.training_status == "Training":
        return "Training already in progress."
    
    try:
        hidden_units = [int(x.strip()) for x in hidden_units_str.split(",")]
    except:
        return "Invalid hidden units format."

    params = {
        'split': split,
        'batch_size': int(batch_size),
        'epochs': int(epochs),
        'lr': float(lr),
        'hidden_units': hidden_units,
        'activation': activation,
        'dropout': dropout,
        'target_fpr': target_fpr
        , 'none_cap': int(none_cap) if none_cap and int(none_cap) > 0 else None
    }
    # Default: do not perform balancing unless requested via UI checkbox
    params['balance'] = bool(balance)
    
    t = threading.Thread(target=worker.train_model, args=(params,))
    t.start()
    return "Training started..."

def get_log_handler():
    return global_state.get_log()

def get_status_handler():
    return global_state.training_status

def get_roc_plot():
    if not global_state.roc_curve:
        return None
    
    roc = global_state.roc_curve
    fprs = [x[2] for x in roc]
    tprs = [x[1] for x in roc]
    
    fig = plt.figure()
    plt.plot(fprs, tprs)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("None-bias ROC")
    plt.grid(True)
    return fig

def download_model_handler():
    model_path, status = worker.save_trained_model()
    return model_path, status

def predict_handler(audio, metadata, det_enabled, det_threshold, det_min_gap, use_manual, threshold):
    if not audio:
        return pd.DataFrame(), "No audio file provided."
    
    # Update detector config
    global_state.detector_config['enabled'] = det_enabled
    global_state.detector_config['threshold'] = det_threshold
    global_state.detector_config['min_gap'] = det_min_gap
    
    thresh = threshold if use_manual else None
    results = worker.predict(audio, threshold_override=thresh, metadata_text=metadata)
    
    # Convert to dataframe
    df = pd.DataFrame(results, columns=["Timestamp (s)", "Label", "Confidence"])
    df["Timestamp (s)"] = df["Timestamp (s)"].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
    df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
    
    status = f"Detected {len(results)} event(s)"
    return df, status

def adjust_threshold_handler(threshold):
    results = worker.adjust_threshold(threshold)
    
    # Convert to dataframe
    df = pd.DataFrame(results, columns=["Timestamp (s)", "Label", "Confidence"])
    df["Timestamp (s)"] = df["Timestamp (s)"].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
    df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
    
    return df

def create_waveform_plot(start_time=None, end_time=None):
    """Create waveform visualization with event markers.
    
    Args:
        start_time: Start time in seconds (None = beginning)
        end_time: End time in seconds (None = end)
    """
    if not global_state.last_audio_data or not global_state.last_inference_results:
        return None
    
    try:
        # Get cached audio data
        wav, sr = global_state.last_audio_data
        duration = len(wav) / sr
        
        # Handle time range
        start_time = start_time or 0.0
        end_time = end_time or duration
        start_time = max(0, min(start_time, duration))
        end_time = max(start_time, min(end_time, duration))
        
        # Extract time range
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        wav_range = wav[start_sample:end_sample]
        time_range = np.arange(len(wav_range)) / sr + start_time
        
        # Downsample waveform for display if longer than 2 minutes
        max_display_samples = sr * 120  # 2 minutes at full resolution
        if len(wav_range) > max_display_samples:
            downsample_factor = int(np.ceil(len(wav_range) / max_display_samples))
            wav_display = wav_range[::downsample_factor]
            time_display = time_range[::downsample_factor]
        else:
            wav_display = wav_range
            time_display = time_range
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(time_display, wav_display, color='steelblue', linewidth=0.5, alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Audio Waveform with Detected Events ({start_time:.1f}s - {end_time:.1f}s)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(start_time, end_time)
        
        # Get unique labels for color mapping
        labels_in_results = list(set(r[1] for r in global_state.last_inference_results))
        colors = plt.cm.tab10(np.linspace(0, 1, len(labels_in_results)))
        label_to_color = {label: colors[i] for i, label in enumerate(labels_in_results)}
        
        # Add event markers (only within time range)
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        label_y = y_max - y_range * 0.1  # Position labels near top
        
        for timestamp, label, confidence in global_state.last_inference_results:
            if not isinstance(timestamp, (int, float)):
                continue
            # Only show events within the selected time range
            if timestamp < start_time or timestamp > end_time:
                continue
                
            color = label_to_color.get(label, 'red')
            
            # Vertical line
            ax.axvline(x=timestamp, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
            
            # Label with confidence
            ax.text(timestamp, label_y, f'{label}\n{confidence:.2f}', 
                    rotation=45, ha='left', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.6))
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=label_to_color[label], label=label) 
                          for label in labels_in_results]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error creating waveform plot: {e}")
        return None

def evaluate_handler(audio_files, metadata_text):
    if not audio_files or not metadata_text:
        return None, pd.DataFrame(), "No test data provided."
    
    # Split metadata by file (assuming one metadata block per file)
    # For simplicity, assume user provides matching order
    metadata_list = [metadata_text] * len(audio_files) if isinstance(audio_files, list) else [metadata_text]
    
    metrics = worker.evaluate_test_set(audio_files if isinstance(audio_files, list) else [audio_files], metadata_list)
    
    if "error" in metrics:
        return None, pd.DataFrame(), metrics["error"]
    
    # Create confusion matrix plot
    cm = np.array(metrics['confusion_matrix'])
    labels = metrics['labels']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')
    
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    
    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    # Create metrics dataframe
    per_class = metrics['per_class_metrics']
    df = pd.DataFrame.from_dict(per_class, orient='index')
    df = df.reset_index().rename(columns={'index': 'Label'})
    
    status = f"Evaluated on {sum(per_class[l]['support'] for l in per_class)} samples"
    
    return fig, df, status

with gr.Blocks(title="A3EM AI Trainer") as demo:
    gr.Markdown("# A3EM AI Few-Shot Trainer")
    
    with gr.Tab("Data Management"):
        with gr.Row():
            with gr.Column():
                file_in = gr.File(label="Audio File(s)", file_count="multiple")
                label_in = gr.Textbox(label="Label")
                is_bg_in = gr.Checkbox(label="Is Background/None")
                meta_in = gr.TextArea(label="Metadata (CSV: timestamp,label)", placeholder="6.725604,Unknown\n6.803292,Fireworks")
                
                with gr.Accordion("Event Detection Settings", open=False):
                    detector_enabled = gr.Checkbox(label="Enable Event Detection", value=True, info="Auto-detect events when no metadata provided")
                    detector_type = gr.Dropdown(["Spectral Flux"], value="Spectral Flux", label="Detector Type")
                    detector_threshold = gr.Slider(1.0, 20.0, value=config.DEFAULT_DETECTOR_THRESHOLD, label="Detection Threshold")
                    detector_min_gap = gr.Slider(0.01, 1.0, value=config.DEFAULT_DETECTOR_MIN_GAP, label="Min Gap Between Events (s)")
                
                upload_btn = gr.Button("Upload & Add")
                upload_msg = gr.Textbox(label="Status")
            
            with gr.Column():
                summary_table = gr.Dataframe(label="Dataset Summary")
                refresh_btn = gr.Button("Refresh Summary")
        
        upload_btn.click(upload_file_handler, 
                        inputs=[file_in, label_in, is_bg_in, meta_in, detector_enabled, detector_type, detector_threshold, detector_min_gap], 
                        outputs=[upload_msg, summary_table])
        refresh_btn.click(get_dataset_summary, outputs=summary_table)

    with gr.Tab("Training Configuration"):
        with gr.Row():
            with gr.Column():
                split_slider = gr.Slider(0.5, 0.95, value=config.DEFAULT_TRAIN_SPLIT, label="Train Split")
                batch_slider = gr.Slider(8, 128, value=config.DEFAULT_BATCH_SIZE, step=8, label="Batch Size")
                epochs_slider = gr.Slider(10, 500, value=config.DEFAULT_EPOCHS, label="Epochs")
                lr_slider = gr.Slider(1e-5, 1e-2, value=config.DEFAULT_LEARNING_RATE, label="Learning Rate")
                balance_checkbox = gr.Checkbox(label="Balance classes (upsample at train time)", value=False)
            
            with gr.Column():
                hidden_in = gr.Textbox(value="256, 128", label="Hidden Units (comma separated)")
                act_drop = gr.Dropdown(["relu", "gelu", "swish"], value="gelu", label="Activation")
                drop_slider = gr.Slider(0.0, 0.5, value=0.25, label="Dropout")
                fpr_slider = gr.Slider(0.01, 0.5, value=0.10, label="Target FPR (for None threshold)")
                none_cap = gr.Number(value=100, label="Max 'None' samples to use (0=disabled)")
        
        train_btn = gr.Button("Start Training", variant="primary")
        status_txt = gr.Textbox(label="Training Status")
        log_txt = gr.TextArea(label="Training Log", lines=10, max_lines=20)
        
        with gr.Row():
            download_btn = gr.Button("Save Trained Model")
            download_file = gr.File(label="Model File", visible=False)
            download_status = gr.Textbox(label="Model Save Status")
            
        train_btn.click(start_training_handler, 
            inputs=[split_slider, batch_slider, epochs_slider, lr_slider, hidden_in, act_drop, drop_slider, fpr_slider, none_cap, balance_checkbox],
            outputs=status_txt)
            
        download_btn.click(download_model_handler, outputs=[download_file, download_status])
        
        # Manual refresh for logs (auto-refresh 'every' kwarg not supported in this version)
        refresh_logs_btn = gr.Button("Refresh Logs & Status")
        refresh_logs_btn.click(get_log_handler, None, log_txt)
        refresh_logs_btn.click(get_status_handler, None, status_txt)

        # Initial load
        demo.load(get_log_handler, None, log_txt)
        demo.load(get_status_handler, None, status_txt)

    with gr.Tab("Inference & Analysis"):
        with gr.Tabs():
            with gr.Tab("Single File Prediction"):
                with gr.Row():
                    with gr.Column():
                        audio_in = gr.File(label="Test Audio File")
                        meta_in_infer = gr.TextArea(
                            label="Manual Event Timestamps (optional)", 
                            placeholder="6.725604\n8.125000\n12.450000",
                            info="One timestamp per line. Leave empty to use detector."
                        )
                        
                        with gr.Accordion("Event Detection Settings", open=False):
                            detector_enabled_infer = gr.Checkbox(label="Enable Event Detection", value=True)
                            detector_threshold_infer = gr.Slider(1.0, 20.0, value=config.DEFAULT_DETECTOR_THRESHOLD, label="Detection Threshold")
                            detector_min_gap_infer = gr.Slider(0.01, 1.0, value=config.DEFAULT_DETECTOR_MIN_GAP, label="Min Gap (s)")
                        
                        use_manual_thresh = gr.Checkbox(label="Use Manual Threshold", value=False)
                        thresh_slider = gr.Slider(-1.0, 1.0, value=0.0, label="None-bias Threshold Override", step=0.01)
                        
                        predict_btn = gr.Button("Predict Events", variant="primary")
                        status_infer = gr.Textbox(label="Status")
                
                # Waveform visualization (full width)
                waveform_plot = gr.Plot(label="Audio Waveform with Detected Events")
                
                with gr.Row():
                    with gr.Column():
                        results_table = gr.Dataframe(
                            label="Event Predictions",
                            headers=["Timestamp (s)", "Label", "Confidence"],
                            interactive=False
                        )
                        
                        gr.Markdown("### Adjust Threshold")
                        thresh_adjust_slider = gr.Slider(
                            -1.0, 1.0, value=0.0, step=0.01,
                            label="Adjust None-bias Threshold",
                            info="Drag to recalculate predictions with cached embeddings"
                        )
                        
                        gr.Markdown("### Zoom Waveform (optional)")
                        with gr.Row():
                            wf_start_slider = gr.Slider(
                                0, 600, value=0, step=0.1,
                                label="Start Time (s)"
                            )
                            wf_end_slider = gr.Slider(
                                0, 600, value=600, step=0.1,
                                label="End Time (s)"
                            )
                        
                        export_btn = gr.Button("Export Results as CSV")
                        export_file = gr.File(label="Download CSV", visible=False)
                
                def predict_with_plot(audio, metadata, det_enabled, det_threshold, det_min_gap, use_manual, threshold):
                    df, status = predict_handler(audio, metadata, det_enabled, det_threshold, det_min_gap, use_manual, threshold)
                    plot = create_waveform_plot()
                    return df, status, plot
                
                predict_btn.click(
                    predict_with_plot,
                    inputs=[audio_in, meta_in_infer, detector_enabled_infer, detector_threshold_infer, 
                            detector_min_gap_infer, use_manual_thresh, thresh_slider],
                    outputs=[results_table, status_infer, waveform_plot]
                )
                
                def adjust_with_plot(threshold, wf_start, wf_end):
                    df = adjust_threshold_handler(threshold)
                    plot = create_waveform_plot(start_time=wf_start, end_time=wf_end)
                    return df, plot
                
                thresh_adjust_slider.change(
                    adjust_with_plot,
                    inputs=[thresh_adjust_slider, wf_start_slider, wf_end_slider],
                    outputs=[results_table, waveform_plot]
                )
                
                # Also update when time range sliders change
                def update_waveform_range(wf_start, wf_end):
                    plot = create_waveform_plot(start_time=wf_start, end_time=wf_end)
                    return plot
                
                wf_start_slider.change(
                    update_waveform_range,
                    inputs=[wf_start_slider, wf_end_slider],
                    outputs=[waveform_plot]
                )
                wf_end_slider.change(
                    update_waveform_range,
                    inputs=[wf_start_slider, wf_end_slider],
                    outputs=[waveform_plot]
                )
                
                def export_results():
                    if not global_state.last_inference_results:
                        return None
                    df = pd.DataFrame(global_state.last_inference_results, 
                                    columns=["Timestamp (s)", "Label", "Confidence"])
                    path = "/tmp/predictions.csv"
                    df.to_csv(path, index=False)
                    return path
                
                export_btn.click(export_results, outputs=[export_file])
            
            with gr.Tab("Test Set Evaluation"):
                with gr.Row():
                    with gr.Column():
                        test_audio_in = gr.File(label="Test Audio File(s)", file_count="multiple")
                        test_meta_in = gr.TextArea(
                            label="Ground Truth Metadata (CSV: timestamp,label)",
                            placeholder="6.725604,Gunshot\n8.125000,Explosion\n12.450000,None",
                            info="Provide ground truth labels for evaluation"
                        )
                        evaluate_btn = gr.Button("Evaluate Model", variant="primary")
                        eval_status = gr.Textbox(label="Status")
                    
                    with gr.Column():
                        confusion_plot = gr.Plot(label="Confusion Matrix")
                        metrics_table = gr.Dataframe(
                            label="Per-Class Metrics",
                            headers=["Label", "Precision", "Recall", "F1", "Support"]
                        )
                
                evaluate_btn.click(
                    evaluate_handler,
                    inputs=[test_audio_in, test_meta_in],
                    outputs=[confusion_plot, metrics_table, eval_status]
                )
            
            with gr.Tab("ROC Analysis"):
                with gr.Row():
                    with gr.Column():
                        refresh_roc = gr.Button("Refresh ROC Curve")
                        gr.Markdown("""
                        **ROC Curve**: Shows the trade-off between True Positive Rate and False Positive Rate
                        for the None-bias threshold. Use this to understand how threshold changes affect
                        None-class detection.
                        """)
                    
                    with gr.Column():
                        roc_plot = gr.Plot(label="ROC Curve")
                
                refresh_roc.click(get_roc_plot, outputs=roc_plot)
                demo.load(get_roc_plot, outputs=roc_plot)

if __name__ == "__main__":
    # Initialize system on startup
    worker.init_system()
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
