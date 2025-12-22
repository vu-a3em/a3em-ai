import gradio as gr
import threading
import time
import pandas as pd
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

def start_training_handler(split, batch_size, epochs, lr, hidden_units_str, activation, dropout, target_fpr, none_cap):
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

def predict_handler(audio, use_manual, threshold):
    thresh = threshold if use_manual else None
    label, conf, _ = worker.predict(audio, thresh)
    return label, conf

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
            
            with gr.Column():
                hidden_in = gr.Textbox(value="256, 128", label="Hidden Units (comma separated)")
                act_drop = gr.Dropdown(["relu", "gelu", "swish"], value="gelu", label="Activation")
                drop_slider = gr.Slider(0.0, 0.5, value=0.25, label="Dropout")
                fpr_slider = gr.Slider(0.01, 0.5, value=0.10, label="Target FPR (for None threshold)")
                none_cap = gr.Number(value=0, label="Max 'None' samples to use (0=disabled)")
        
        train_btn = gr.Button("Start Training", variant="primary")
        status_txt = gr.Textbox(label="Training Status")
        log_txt = gr.TextArea(label="Training Log", lines=10, max_lines=20)
            
        train_btn.click(start_training_handler, 
                        inputs=[split_slider, batch_slider, epochs_slider, lr_slider, hidden_in, act_drop, drop_slider, fpr_slider, none_cap],
                        outputs=status_txt)
            
        # Manual refresh for logs (auto-refresh 'every' kwarg not supported in this version)
        refresh_logs_btn = gr.Button("Refresh Logs & Status")
        refresh_logs_btn.click(get_log_handler, None, log_txt)
        refresh_logs_btn.click(get_status_handler, None, status_txt)

        # Initial load
        demo.load(get_log_handler, None, log_txt)
        demo.load(get_status_handler, None, status_txt)

    with gr.Tab("Inference & Analysis"):
        with gr.Row():
            with gr.Column():
                audio_in = gr.File(label="Test Audio")
                use_manual_thresh = gr.Checkbox(label="Use Manual Threshold", value=False)
                thresh_slider = gr.Slider(-1.0, 1.0, value=0.0, label="None-bias Threshold Override")
                predict_btn = gr.Button("Predict")
            
            with gr.Column():
                pred_label = gr.Textbox(label="Predicted Label")
                pred_conf = gr.Textbox(label="Confidence")
                roc_plot = gr.Plot(label="ROC Curve")
                refresh_roc = gr.Button("Refresh ROC")
        
        predict_btn.click(predict_handler, inputs=[audio_in, use_manual_thresh, thresh_slider], outputs=[pred_label, pred_conf])
        refresh_roc.click(get_roc_plot, outputs=roc_plot)

if __name__ == "__main__":
    # Initialize system on startup
    worker.init_system()
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
