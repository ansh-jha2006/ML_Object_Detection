import os
#%%
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import tensorflow as tf
import tensorflow_datasets as tfds
#%%
# Global constants and settings
im_width = 75
im_height = 75
use_normalized_coordinates = True
#%%
# Helper functions for drawing bounding boxes
def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color="red", thickness=1, display_string=None, use_normalized_coordinates=True):
    draw = PIL.ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)

def draw_bounding_boxes_on_image(image, boxes, color=[], thickness=1, display_str_list=()):
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N,4]')
    for i in range(boxes_shape[0]):
        draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 3], boxes[i, 2], color=color[i] if color else 'red', thickness=thickness, display_string=display_str_list[i] if display_str_list else None)

def draw_bounding_boxes_on_image_array(image, boxes, color=[], thickness=1, display_str_list=()):
    image_pil = PIL.Image.fromarray(image)
    rgbimg = PIL.Image.new("RGBA", image_pil.size)
    rgbimg.paste(image_pil)
    draw_bounding_boxes_on_image(rgbimg, boxes, color, thickness, display_str_list)
    return np.array(rgbimg)
#%%
# Utility function for converting datasets to numpy arrays
def dataset_to_numpy_util(training_dataset, validation_dataset, N):
    if tf.executing_eagerly():
        for validation_digits, (validation_labels, validation_bboxes) in validation_dataset.take(1):
            validation_digits = validation_digits.numpy()
            validation_labels = validation_labels.numpy()
            validation_bboxes = validation_bboxes.numpy()
        for training_digits, (training_labels, training_bboxes) in training_dataset.take(1):
            training_digits = training_digits.numpy()
            training_labels = training_labels.numpy()
            training_bboxes = training_bboxes.numpy()
            
    validation_labels = np.argmax(validation_labels, axis=1)
    training_labels = np.argmax(training_labels, axis=1)
    return (training_digits, training_labels, training_bboxes,
            validation_digits, validation_labels, validation_bboxes)
#%%
# Helper function for creating digits from local fonts
MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")
def create_digits_from_local_fonts(n):
    font_labels = []
    img = PIL.Image.new('LA', (75*n, 75), color=(0,255))
    font1 = PIL.ImageFont.truetype(os.path.join(MATPLOTLIB_FONT_DIR, 'DejaVuSansMono-Oblique.ttf'), 25)
    font2 = PIL.ImageFont.truetype(os.path.join(MATPLOTLIB_FONT_DIR, 'STIXGeneral.ttf'), 25)
    d = PIL.ImageDraw.Draw(img)

    for i in range(n):
        font_labels.append(i%10)
        d.text((75*i+75, 0 if i < 10 else -4), str(i%10), fill=(255,255), font=font1 if i < 10 else font2)

    font_digits = np.array(img.getdata(), np.float32)[:, 0] / 255.0
    font_digits = np.reshape(np.stack(np.split(np.reshape(font_digits, [75, 75*n]), n, axis=1), axis=0), [n, 75, 75])
    return font_digits, font_labels
#%%
# Function for displaying digits with bounding boxes
def display_digits_with_boxes(digits, predictions, labels, pred_bboxes, bboxes, iou, title):
    n = 10
    indexes = np.random.choice(len(predictions), size=n)
    n_digits = digits[indexes]
    n_predictions = predictions[indexes]
    n_labels = labels[indexes]

    n_iou = []
    if len(iou) > 0:
        n_iou = iou[indexes]
    
    if len(pred_bboxes) > 0:
        n_pred_bboxes = pred_bboxes[indexes]
        
    if len(bboxes) > 0:
        n_bboxes = bboxes[indexes]

    n_digits = n_digits * 255.0
    n_digits = n_digits.reshape(n, 75, 75)

    fig = plt.figure(figsize=(20,4))
    plt.title(title)
    plt.xticks([])
    plt.yticks([])

    for i in range(n):
        ax = fig.add_subplot(1, n, i + 1)
        bboxes_to_plot = []
        
        if len(pred_bboxes) > 0:
            bboxes_to_plot.append(n_pred_bboxes[i])
            
        if len(bboxes) > 0:
            bboxes_to_plot.append(n_bboxes[i])

        img_to_draw = draw_bounding_boxes_on_image_array(image=n_digits[i],
                                                         boxes=np.asarray(bboxes_to_plot),
                                                         color=['red', 'green'],
                                                         display_str_list=["True", "Pred"])

        plt.xlabel(n_predictions[i])
        plt.xticks([])
        plt.yticks([])
        
        if n_predictions[i] != n_labels[i]:
            ax.xaxis.label.set_color('red')

        plt.imshow(img_to_draw)
        iou_threshold = 0.5
        if len(iou) > i:
            color = "black"
            if n_iou[i][0] < iou_threshold:
                color = "red"
            ax.text(0.2, -0.3, "iou: %s" %(n_iou[i][0]), color=color, transform=ax.transAxes)
    plt.show()
#%%
# TensorFlow dataset functions
strategy = tf.distribute.get_strategy()
BATCH_SIZE = 64 * strategy.num_replicas_in_sync
#%%
def read_image_tfds(image, label):
    xmin = tf.random.uniform([], 0, 48, dtype=tf.int32)
    ymin = tf.random.uniform([], 0, 48, dtype=tf.int32)
    image = tf.reshape(image, (28, 28, 1))
    image = tf.image.pad_to_bounding_box(image, ymin, xmin, 75, 75)
    image = tf.cast(image, tf.float32) / 255.0
    xmin = tf.cast(xmin, tf.float32)
    ymin = tf.cast(ymin, tf.float32)

    xmax = (xmin + 28) / 75
    ymax = (ymin + 28) / 75
    xmin = xmin / 75
    ymin = ymin / 75
    
    return image, (tf.one_hot(label, 10), [xmin, ymin, xmax, ymax])
#%%
def get_training_dataset():
    with strategy.scope():
        dataset = tfds.load("mnist", split="train", as_supervised=True, try_gcs=True)
        dataset = dataset.map(read_image_tfds, num_parallel_calls=16)
        dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        dataset = dataset.prefetch(-1)
        return dataset
#%%
def get_validation_dataset():
    with strategy.scope():
        dataset = tfds.load("mnist", split="train", as_supervised=True, try_gcs=True)
        dataset = dataset.map(read_image_tfds, num_parallel_calls=16)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        dataset = dataset.repeat()
        return dataset
#%%
# Model Building, Training, and Visualization
with strategy.scope():
    # Define the model architecture
    input_tensor = tf.keras.layers.Input(shape=(75, 75, 1))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    # Output layers for classification and regression
    class_output = tf.keras.layers.Dense(10, activation='softmax', name='class_output')(x)
    box_output = tf.keras.layers.Dense(4, activation='sigmoid', name='box_output')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=[class_output, box_output])
#%%
# Compile the model with appropriate losses and metrics
model.compile(optimizer='adam',
              loss={'class_output': 'categorical_crossentropy',
                    'box_output': 'mse'},
              metrics={'class_output': 'accuracy',
                       'box_output': 'mse'})

#%%
# Get training and validation datasets
training_dataset = get_training_dataset()
validation_dataset = get_validation_dataset()
#%%
# Train the model
print("Training the model...")
model.fit(
    training_dataset,
    epochs=5,
    steps_per_epoch=500,
    validation_data=validation_dataset,
    validation_steps=10
)
#%%
# Get validation data for visualization
print("Getting validation data for visualization...")
for validation_digits, (validation_labels, validation_bboxes) in validation_dataset.take(1):
    validation_digits = validation_digits.numpy()
    validation_labels = np.argmax(validation_labels.numpy(), axis=1)
    validation_bboxes = validation_bboxes.numpy()
#%%
# Generate predictions
print("Generating predictions...")
predictions = model.predict(validation_digits)
predicted_labels = np.argmax(predictions[0], axis=1)
predicted_bboxes = predictions[1]
#%%
# Display results
print("Displaying results...")
display_digits_with_boxes(
    validation_digits,
    predicted_labels,
    validation_labels,
    predicted_bboxes,
    validation_bboxes,
    np.array([]), # IoU calculation is a separate step not included here
    "Validation Results (True vs. Predicted BBoxes)"
)
#%%