from microesc.datasets.RavenProDataSet import RavenProDataSet
from microesc.classification.Resnet50 import create_resnet50_model, Resnet50Params
from microesc import keras
import microesc.tools as tools

# Generate the ResNet50 model
params = Resnet50Params()
params.num_classes = 4
model = create_resnet50_model(params, load_pretrained_weights=False, freeze_pretrained_layers=False)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),  # type: ignore
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
model.summary()

# Create the full audio dataset and split it into a training and testing dataset
audio_path = '/Volumes/AIDataSets/DataSet/audio'
annotation_path = '/Volumes/AIDataSets/DataSet/annotations'
dataset = RavenProDataSet(audio_path, annotation_path, target_sample_rate_hz=8000, target_clip_length=3.0, training_split_percent=0.8, uniform_classes_per_batch=False)
train_ds = dataset.train_dataset(batch_size=32)
test_ds = dataset.test_dataset(batch_size=32)
dataset.summary()

# Train and save the best model
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history: keras.callbacks.History = model.fit(train_ds, epochs=10000, validation_data=test_ds, callbacks=[callback], verbose=2)  # type: ignore
model.evaluate(test_ds, return_dict=True)
tools.plot_training_history(history)
tools.plot_confusion_matrix(model, test_ds, dataset.idx_to_label)
model.save('resnet50.keras')
