"""Quantization-aware training utilities for model optimization."""

from typing import Optional
import tensorflow_model_optimization as tfmot
from .. import keras
from ..classification.Yamnet import WaveformToLogMel


def replace_groupnorm_with_layernorm(layer: keras.layers.Layer) -> keras.layers.Layer:
    """Replace GroupNormalization with LayerNormalization for TFLite compatibility.
    
    GroupNorm causes issues with TFLite's BROADCAST_TO operation, so we replace
    it with LayerNormalization which is well-supported.
    
    Args:
        layer: Keras layer to potentially replace
    
    Returns:
        LayerNormalization if input was GroupNorm, otherwise returns clone of original layer
    """
    if isinstance(layer, keras.layers.GroupNormalization):
        config = layer.get_config()
        return keras.layers.LayerNormalization(
            axis=-1,  # Normalize over the channel dimension
            epsilon=config.get('epsilon', 1e-3),
            name=layer.name + "_as_layernorm"
        )
    return layer.__class__.from_config(layer.get_config())


def apply_selective_quantization(layer: keras.layers.Layer) -> keras.layers.Layer:
    """Apply quantization annotation to supported layer types.
    
    Quantizes Dense, Conv2D, and DepthwiseConv2D layers while skipping
    custom layers (like WaveformToLogMel) and normalization layers.
    
    Args:
        layer: Keras layer to potentially quantize
    
    Returns:
        Quantization-annotated layer or original layer
    """
    # Skip custom layers and normalization layers (not supported by tfmot)
    if isinstance(layer, (WaveformToLogMel, keras.layers.LayerNormalization)):
        return layer
    
    # Quantize only these specific layer types
    quantize_types = (keras.layers.Dense, keras.layers.Conv2D, keras.layers.DepthwiseConv2D)
    if isinstance(layer, quantize_types):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    
    # Return all other layers unchanged
    return layer


def prepare_model_for_quantization(model: keras.Model, 
                                   verbose: bool = True) -> keras.Model:
    """Prepare a model for quantization-aware training.
    
    This function performs the complete workflow:
    1. Replace GroupNorm with LayerNorm for TFLite compatibility
    2. Copy weights from original model
    3. Apply selective quantization annotations
    4. Apply quantization to create QAT model
    
    Args:
        model: Original trained model
        verbose: Whether to print progress messages
    
    Returns:
        Quantization-aware training model ready for fine-tuning
    """
    if verbose:
        print("\n=== PREPARING MODEL FOR QUANTIZATION ===")
        print("Step 1: Replacing GroupNorm with LayerNorm...")
    
    # Step 1: Replace GroupNorm with LayerNorm
    model_with_layernorm = keras.models.clone_model(
        model, 
        clone_function=replace_groupnorm_with_layernorm
    )
    
    # Copy weights carefully - GroupNorm and LayerNorm have different structures
    if verbose:
        print("Step 2: Copying weights...")
    for orig_layer, new_layer in zip(model.layers, model_with_layernorm.layers):
        if isinstance(orig_layer, (keras.layers.GroupNormalization, WaveformToLogMel)):
            # For GroupNorm, we can't directly copy weights to LayerNorm
            # The LayerNorm will be initialized randomly but fine-tuned during QAT
            continue
        orig_weights = orig_layer.get_weights()
        if orig_weights:
            new_layer.set_weights(orig_weights)
    
    # Step 2: Apply selective quantization
    if verbose:
        print("Step 3: Applying quantization annotations...")
    annotated_model = keras.models.clone_model(
        model_with_layernorm, 
        clone_function=apply_selective_quantization
    )
    
    # Copy weights to annotated model
    for orig_layer, annot_layer in zip(model_with_layernorm.layers, annotated_model.layers):
        orig_weights = orig_layer.get_weights()
        if not orig_weights:
            continue
        try:
            # For annotated layers, set weights on the inner layer
            if hasattr(annot_layer, 'layer'):
                annot_layer.layer.set_weights(orig_weights)
            else:
                annot_layer.set_weights(orig_weights)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not copy weights for {orig_layer.name}: {e}")
    
    # Step 3: Apply full quantization
    if verbose:
        print("Step 4: Applying quantization...")
    with tfmot.quantization.keras.quantize_scope({'WaveformToLogMel': WaveformToLogMel}):
        quant_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    
    if verbose:
        print("✓ Quantization preparation complete")
    
    return quant_model


def quantize_and_train(model: keras.Model,
                      train_ds,
                      test_ds,
                      epochs: int = 10000,
                      patience: int = 10,
                      learning_rate: float = 1e-4,
                      verbose: int = 2) -> keras.Model:
    """Complete quantization-aware training workflow.
    
    Args:
        model: Original trained model
        train_ds: Training dataset
        test_ds: Validation dataset
        epochs: Maximum training epochs
        patience: Early stopping patience
        learning_rate: Learning rate for QAT
        verbose: Verbosity level (0, 1, or 2)
    
    Returns:
        Quantized and fine-tuned model
    """
    # Prepare model for quantization
    quant_model = prepare_model_for_quantization(model, verbose=(verbose > 0))
    
    # Compile
    quant_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )
    
    # Train with early stopping
    callback = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=patience, 
        restore_best_weights=True
    )
    
    if verbose > 0:
        print("\n=== QUANTIZATION-AWARE TRAINING ===")
    
    history = quant_model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        callbacks=[callback],
        verbose=verbose
    )
    
    return quant_model
