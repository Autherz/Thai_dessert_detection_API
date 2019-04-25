
# Import module and libraries
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger, Callback
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
import math
from models.keras_ssd300 import ssd_300
from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from keras_preprocessing import image

import time
start = time.time()
K.clear_session()

# https://www.ir.com/blog/visualizing-outputs-cnn-model-training-phase
def get_kernels_output(model,
                       layer_index,
                       model_input,
                       training_flag = True):
    get_outputs = K.function([model.layers[0].input,
                              K.learning_phase()],
                             [model.layers[layer_index].output])
    print(model_input.shape)
    kernels_output = get_outputs([
                                model_input,
                                training_flag])[0]



    return kernels_output


def combine_output_images(kernels_output):
    output_count = kernels_output.shpae[1]
    width = int(math.sqrt(output_count))
    height = int(math.ceil(float(output_count)/width))

    if len(kernels_output.shape) == 4:
        output_shape = kernels_output.shape[2:]
        image = np.zeros((height * output_shape[0], width * output_shape[1]),
                         dtype= kernels_output.dtype)

        for index, output in enumerate(kernels_output[0]):
            i = int(index / width)
            j = index % width
            image[i * output_shape[0]: (i + 1) * output_shape[0],
                  j * output_shape[1]: (j + 1) * output_shape[1]] = output
        return image


def get_model_layers_output_combined_image(model,
                                           model_input,
                                           training_flag = True):

    for layer_idx in range(len(model.layers)):
        kernels_output = get_kernels_output(model, layer_idx,
                                            model_input,
                                            training_flag)

        combined_outputs_image = combine_output_images(kernels_output)

        if combined_outputs_image is not None:
            print(model.layers[layer_idx].name)
            plt.matshow(combined_outputs_image.T, vmin = 0.0, vmax = 1.0)
            plt.show()


class OutputImages(Callback):

    def __init__(self, input_data):
        self.input_data = input_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 50 == 0:
            get_model_layers_output_combined_image(self.model,
                                                   self.input_data)

# ---------------------------------------------------------------------------------------------------------------------

# Set Model configuration


img_height = 300  # Height of the the model input image
img_width = 300  # Width
img_channels = 3  # Number of color channels of the model input images
mean_color = [123, 117, 104]  # The per-channel mean of the images in the dataset. Do not change this value # if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0]  # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 17  # Number of positive classes
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
# scales_pascal = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]  # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters

two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300]  # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2]  # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True

# End Setting

# Build Model

# 1: Build the Keras model.

K.clear_session()  # Clear previous models from memory.

model = ssd_300(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='training',
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=mean_color,
                swap_channels=swap_channels)

print(model.summary())

# 2: Load some weights into the model.

# TODO: Set the path to the weights you want to load.
weights_path = 'model_2.h5'
weights_path = 'ssd300_Thai_dessert_16_epoch-236_loss-3.9260_val_loss-3.6209.h5'

# model.load_weights(weights_path, by_name=True)

# 3: Instantiate an optimizer and the SSD loss function and compile the model.
#    If you want to follow the original Caffe implementation, use the preset SGD
#    optimizer, otherwise I'd recommend the commented-out Adam optimizer.

adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss, metrics=['accuracy'])

'''

# TODO: Set the path to the `.h5` file of the model to be loaded.
model_path = 'model_2.h5'

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

K.clear_session() # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'compute_loss': ssd_loss.compute_loss})

'''

# ---------------------------------------------------------------------------------------------------------------------

# Set Data Generators

# 1: Instantiate two `DataGenerator` objects: One for training, one for validation.

# Optional: If you have enough memory, consider loading the images into memory for the reasons explained above.

train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

# 2: Parse the image and label lists for the training and validation datasets. This can take a while.

# TODO: Set the paths to the datasets here.

'''

# The directories that contain the images.
VOC_2007_images_dir = 'datasets/VOCdevkit/VOC2007/JPEGImages/'
VOC_2012_images_dir = 'datasets/VOCdevkit/VOC2012/JPEGImages/'

# The directories that contain the annotations.
VOC_2007_annotations_dir = 'datasets/VOCdevkit/VOC2007/Annotations/'
VOC_2012_annotations_dir = 'datasets/VOCdevkit/VOC2012/Annotations/'

# The paths to the image sets.
VOC_2007_train_image_set_filename = 'datasets/VOCdevkit/VOC2007/ImageSets/Main/train.txt'
VOC_2012_train_image_set_filename = 'datasets/VOCdevkit/VOC2012/ImageSets/Main/train.txt'
VOC_2007_val_image_set_filename = 'datasets/VOCdevkit/VOC2007/ImageSets/Main/val.txt'
VOC_2012_val_image_set_filename = 'datasets/VOCdevkit/VOC2012/ImageSets/Main/val.txt'
VOC_2007_trainval_image_set_filename = 'datasets/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
VOC_2012_trainval_image_set_filename = 'datasets/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
VOC_2007_test_image_set_filename = 'datasets/VOCdevkit/VOC2007/ImageSets/Main/test.txt'

# cat and dog
VOC_2007_images_dir
VOC_2007_annotations_dir
VOC_2007_test_image_set_filename


# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

train_dataset.parse_xml(images_dirs=[VOC_2007_images_dir,
                                     VOC_2012_images_dir],
                        image_set_filenames=[VOC_2007_trainval_image_set_filename,
                                             VOC_2012_trainval_image_set_filename],
                        annotations_dirs=[VOC_2007_annotations_dir,
                                          VOC_2012_annotations_dir],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False)

val_dataset.parse_xml(images_dirs=[VOC_2007_images_dir],
                      image_set_filenames=[VOC_2007_test_image_set_filename],
                      annotations_dirs=[VOC_2007_annotations_dir],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=True,
                      ret=False)

'''
'''
# cat and dog

VOC_2007_images_dir = 'data/train_od/'
VOC_2007_annotations_dir = 'data/annotation'
VOC_2007_trainval_image_set_filename = 'data/trainval.txt'
VOC_2007_test_image_set_filename = 'data/test.txt'

classes = ['background', 'cat', 'dog']

'''

VOC_2007_images_dir = 'data/train_od_td/'
VOC_2007_annotations_dir = 'data/annotation_td/'
VOC_2007_trainval_image_set_filename = 'data/trainval_td.txt'
VOC_2007_test_image_set_filename = 'data/test_td.txt'

classes = ['background',
           'kanom_bueng',  # 1
           'ba_bin',  # 2
           'Foi_Tong',  # 3
           'grilled_sticky_rice',  # 4
           'Jamongkut',  # 5
           'kanomkrok',  # 6
           'kanom_tan',  # 7
           'krayasat',  # 8
           'layer_sweet_cake',  # 9
           'med_kanoon',  # 10
           'tako',  # 11
           'Thong_ek',  # 12
           'Thong_yib',  # 13
           'Thong_yod',  # 14
           'thai_mango_sticky_rice',  # 15
           'ขนมไทยเสน่ห์จันทร์',  # 16
           'kanom_tuay_fu',  # 17
           ]


train_dataset.parse_xml(images_dirs=[VOC_2007_images_dir],
                        image_set_filenames=[VOC_2007_trainval_image_set_filename],
                        annotations_dirs=[VOC_2007_annotations_dir],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False)

val_dataset.parse_xml(images_dirs=[VOC_2007_images_dir],
                      image_set_filenames=[VOC_2007_test_image_set_filename],
                      annotations_dirs=[VOC_2007_annotations_dir],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=True,
                      ret=False)


# Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
# speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
# option in the constructor, because in that cas the images are in memory already anyway. If you don't
# want to create HDF5 datasets, comment out the subsequent two function calls.

'''
train_dataset.create_hdf5_dataset(file_path='dataset_pascal_voc_07+12_trainval.h5',
                                  resize=False,
                                  variable_image_size=True,
                                  verbose=True)

val_dataset.create_hdf5_dataset(file_path='dataset_pascal_voc_07_test.h5',
                                resize=False,
                                variable_image_size=True,
                                verbose=True)

'''

# 3: Set the batch size.

batch_size = 10  # Change the batch size if you like, or if you run into GPU memory issues.

# 4: Set the image transformations for pre-processing and data augmentation options.

# For the training generator:
ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                            img_width=img_width,
                                            background=mean_color)

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.5,
                                    normalize_coords=normalize_coords)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels,
                                                      resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

# ---------------------------------------------------------------------------------------------------------------------

# Set remaining training parameter s

# Define a learning rate schedule.


def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001


# Define model callbacks.

# TODO: Set the filepath under which you want to save the model.
model_checkpoint = ModelCheckpoint(filepath='ssd300_Thai_dessert_16_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)
# model_checkpoint.best =

csv_logger = CSVLogger(filename='ssd300_Thai_D_16_training_log.csv',
                       separator=',',
                       append=True)

learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

terminate_on_nan = TerminateOnNaN()

input_image = image.load_img('data/train_od_td/Foi_Tong_0.jpg', target_size=(300, 300))
input_image = image.img_to_array(input_image)
print(input_image.shape)

output_images = OutputImages(input_image)

callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan,
             #output_images
             ]

# ---------------------------------------------------------------------------------------------------------------------

# Train
# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch = 0
final_epoch = 100
steps_per_epoch = 1165 // batch_size


history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size//batch_size),
                              initial_epoch=initial_epoch,
                                )
end = time.time()
print(end - start)
# ---------------------------------------------------------------------------------------------------------------------

# Make Prediction

# 1: Set the generator for the predictions.

predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=True,
                                         transformations=[convert_to_3_channels,
                                                          resize],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'filenames',
                                                  'inverse_transform',
                                                  'original_images',
                                                  'original_labels'},
                                         keep_images_without_gt=False)

# 2: Generate samples.

batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)

i = 0  # Which batch item to look at

print("Image:", batch_filenames[i])
print()
print("Ground truth boxes:\n")
print(np.array(batch_original_labels[i]))

# 3: Make predictions.

y_pred = model.predict(batch_images)

print(model.summary())
def get_activations(model, X_batch):

    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.get_layer('conv9_2_mbox_conf').output])
    activations = get_activations([X_batch, 0])
    return activations

print(get_activations(model, batch_images)[0].shape)

# plt.imshow(get_activations(model, batch_images)[0][0, :, :, 0])

row_size = 8
col_size = 8
activation_index=0
fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
for row in range(0 ,row_size):
      for col in range(0,col_size):
        ax[row][col].imshow(get_activations(model, batch_images)[0][0, :, :, activation_index], )
        activation_index += 1

# weights_conv = model.get_layer('conv1_1').get_weights()[0]  # weight each layer


# 4: Decode the raw predictions in `y_pred`.
y_pred_decoded = decode_detections(y_pred,
                                   confidence_thresh=0.2,
                                   iou_threshold=0.2,
                                   top_k=200,
                                   normalize_coords=normalize_coords,
                                   img_height=img_height,
                                   img_width=img_width)

# 5: Convert the predictions for the original image.

y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)  # แปลงภาพกลับเป็นแบบ เดิม
print(y_pred_decoded_inv)
np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_decoded_inv[i])

# 5: Draw the predicted boxes onto the image

# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
'''
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
'''
# classes = ['background', 'Foi_Tong', 'grilled_sticky_rice' , 'kanomkrok', 'kanom_tan','thai_mango_sticky_rice', 'Thong_yib', 'Thong_yod']

plt.figure(figsize=(20, 12))
plt.imshow(batch_original_images[i])

current_axis = plt.gca()

for box in batch_original_labels[i]:
    xmin = box[1]
    ymin = box[2]
    xmax = box[3]
    ymax = box[4]
    label = '{}'.format(classes[int(box[0])])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

for box in y_pred_decoded_inv[i]:
    xmin = box[2]
    ymin = box[3]
    xmax = box[4]
    ymax = box[5]
    color = colors[int(box[0])]
    label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))
    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

plt.show()










