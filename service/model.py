from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from ssd_encoder_decoder.ssd_output_decoder import decode_detections
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from data_generator.object_detection_2d_image_boxes_validation_utils import BoxFilter
from keras.optimizers import Adam, SGD
import tensorflow as tf

import numpy as np
from copy import deepcopy
from PIL import Image
import inspect
import warnings
import matplotlib.pyplot as plt
import cv2
from keras import backend as K
import gc

image_size = (300, 300, 3)
img_height, img_width, img_channel = image_size[0], image_size[1], image_size[2]

n_classes = 17
scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
aspect_ratios = [[1.0, 2.0, 0.5],
				 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
				 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
				 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
				 [1.0, 2.0, 0.5],
				 [1.0, 2.0, 0.5]]  # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100,300]  # The space between two adjacent anchor box center points for each predictor layer.

offsets = [0.5, 0.5, 0.5, 0.5, 0.5,
		   0.5]  # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.

variances = [0.1, 0.1, 0.2,0.2]  # The variances by which the encoded target coordinates are divided as in the original implementation
mean_color = [123, 117,104]  # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1,0]  # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
clip_boxes = False
normalize_coords = True

model = ssd_300(image_size=image_size,
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

weights_path = 'ssd300_Thai_dessert_16_epoch-01_loss-3.5501_val_loss-3.7872.h5'
model.load_weights(weights_path, by_name=True)


adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss, metrics=['accuracy'])
graph = tf.get_default_graph()

class Data_Generator:

    def __init__(self, image, image_path, labels_format):

        self.labels = None
        self.images = image
        self.filenames = None
        self.eval_neutral = None
        self.image_ids = None
        self.labels_format = None
        self.dataset_size = 1
        self.image_path = image_path
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)

    def generate(self,
                 batch_size=32,
                 transformations=[],
                 label_encoder=None,
                 returns={'processed_images', 'encoded_labels'},
                 keep_images_without_gt=False,
                 degenerate_box_handling='remove'):

        if degenerate_box_handling == 'remove':
            box_filter = BoxFilter(check_overlap=False,
                                   check_min_area=False,
                                   check_degenerate=True,
                                   labels_format=self.labels_format)

        # Override the labels formats of all the transformations to make sure they are set correctly.
        if not (self.labels is None):
            for transform in transformations:
                transform.labels_format = self.labels_format

        #############################################################################################
        # Generate mini batches.
        #############################################################################################

        current = 0

        while True:

            batch_X, batch_y = [], []

            if current >= self.dataset_size:
                current = 0


            #########################################################################################
            # Get the images, (maybe) image IDs, (maybe) labels, etc. for this batch.
            #########################################################################################

            # We prioritize our options in the following order:
            # 1) If we have the images already loaded in memory, get them from there.
            # 2) Else, if we have an HDF5 dataset, get the images from there.
            # 3) Else, if we have neither of the above, we'll have to load the individual image
            #    files from disk.
            batch_indices = self.dataset_indices[current:current+batch_size]
            batch_indices = range(1)
            if not (self.images is None):

                batch_X.append(self.images)

                if not (self.filenames is None):
                    batch_filenames = self.filenames[current:current+batch_size]
                else:
                    batch_filenames = None
            else:
                print("Images is None")
                batch_filenames = self.filenames[current:current+batch_size]
                for filename in batch_filenames:
                    with Image.open(filename) as image:
                        batch_X.append(np.array(image, dtype=np.uint8))

            if 'original_images' in returns:
                batch_original_images = deepcopy(batch_X) # The original, unaltered images
            if 'original_labels' in returns:
                batch_original_labels = deepcopy(batch_y) # The original, unaltered labels

            current += batch_size

            #########################################################################################
            # Maybe perform image transformations.
            #########################################################################################

            batch_items_to_remove = [] # In case we need to remove any images from the batch, store their indices in this list.
            batch_inverse_transforms = []
            for i in range(len(batch_X)):

                if not (self.labels is None):
                    # Convert the labels for this image to an array (in case they aren't already).
                    batch_y[i] = np.array(batch_y[i])
                    # If this image has no ground truth boxes, maybe we don't want to keep it in the batch.
                    if (batch_y[i].size == 0) and not keep_images_without_gt:
                        batch_items_to_remove.append(i)
                        batch_inverse_transforms.append([])
                        continue

                # Apply any image transformations we may have received.
                if transformations:

                    inverse_transforms = []

                    for transform in transformations:
                        # print(transform)
                        if not (self.labels is None):

                            if ('inverse_transform' in returns) and ('return_inverter' in inspect.signature(transform).parameters):
                                batch_X[i], batch_y[i], inverse_transform = transform(batch_X[i], batch_y[i], return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i], batch_y[i] = transform(batch_X[i], batch_y[i])

                            if batch_X[i] is None: # In case the transform failed to produce an output image, which is possible for some random transforms.
                                batch_items_to_remove.append(i)
                                batch_inverse_transforms.append([])
                                continue

                        else:

                            if ('inverse_transform' in returns) and ('return_inverter' in inspect.signature(transform).parameters):
                                batch_X[i], inverse_transform = transform(batch_X[i], return_inverter=True)
                                inverse_transforms.append(inverse_transform)
                            else:
                                batch_X[i] = transform(batch_X[i])

                    batch_inverse_transforms.append(inverse_transforms[::-1])

                #########################################################################################
                # Check for degenerate boxes in this batch item.
                #########################################################################################

                if not (self.labels is None):

                    xmin = self.labels_format['xmin']
                    ymin = self.labels_format['ymin']
                    xmax = self.labels_format['xmax']
                    ymax = self.labels_format['ymax']

                    if np.any(batch_y[i][:,xmax] - batch_y[i][:,xmin] <= 0) or np.any(batch_y[i][:,ymax] - batch_y[i][:,ymin] <= 0):
                        if degenerate_box_handling == 'warn':
                            warnings.warn("Detected degenerate ground truth bounding boxes for batch item {} with bounding boxes {}, ".format(i, batch_y[i]) +
                                          "i.e. bounding boxes where xmax <= xmin and/or ymax <= ymin. " +
                                          "This could mean that your dataset contains degenerate ground truth boxes, or that any image transformations you may apply might " +
                                          "result in degenerate ground truth boxes, or that you are parsing the ground truth in the wrong coordinate format." +
                                          "Degenerate ground truth bounding boxes may lead to NaN errors during the training.")
                        elif degenerate_box_handling == 'remove':
                            batch_y[i] = box_filter(batch_y[i])
                            if (batch_y[i].size == 0) and not keep_images_without_gt:
                                batch_items_to_remove.append(i)

            #########################################################################################
            # Remove any items we might not want to keep from the batch.
            #########################################################################################

            if batch_items_to_remove:
                for j in sorted(batch_items_to_remove, reverse=True):
                    # This isn't efficient, but it hopefully shouldn't need to be done often anyway.
                    batch_X.pop(j)
                    batch_filenames.pop(j)
                    if batch_inverse_transforms: batch_inverse_transforms.pop(j)
                    if not (self.labels is None): batch_y.pop(j)
                    # if not (self.image_ids is None): batch_image_ids.pop(j)
                    # if not (self.eval_neutral is None): batch_eval_neutral.pop(j)
                    if 'original_images' in returns: batch_original_images.pop(j)
                    if 'original_labels' in returns and not (self.labels is None): batch_original_labels.pop(j)

            #########################################################################################

            # CAUTION: Converting `batch_X` into an array will result in an empty batch if the images have varying sizes
            #          or varying numbers of channels. At this point, all images must have the same size and the same
            #          number of channels.
            batch_X = np.array(batch_X)
            if (batch_X.size == 0):
                raise DegenerateBatchError("You produced an empty batch. This might be because the images in the batch vary " +
                                           "in their size and/or number of channels. Note that after all transformations " +
                                           "(if any were given) have been applied to all images in the batch, all images " +
                                           "must be homogenous in size along all axes.")

            #########################################################################################
            # If we have a label encoder, encode our labels.
            #########################################################################################

            if not (label_encoder is None or self.labels is None):

                if ('matched_anchors' in returns) and isinstance(label_encoder, SSDInputEncoder):
                    batch_y_encoded, batch_matched_anchors = label_encoder(batch_y, diagnostics=True)
                else:
                    batch_y_encoded = label_encoder(batch_y, diagnostics=False)
                    batch_matched_anchors = None

            else:
                batch_y_encoded = None
                batch_matched_anchors = None

            #########################################################################################
            # Compose the output.
            #########################################################################################

            ret = []
            if 'processed_images' in returns: ret.append(batch_X)
            if 'encoded_labels' in returns: ret.append(batch_y_encoded)
            if 'matched_anchors' in returns: ret.append(batch_matched_anchors)
            if 'processed_labels' in returns: ret.append(batch_y)
            if 'filenames' in returns: ret.append(batch_filenames)
            # if 'image_ids' in returns: ret.append(batch_image_ids)
            # if 'evaluation-neutral' in returns: ret.append(batch_eval_neutral)
            if 'inverse_transform' in returns: ret.append(batch_inverse_transforms)
            if 'original_images' in returns: ret.append(batch_original_images)
            if 'original_labels' in returns: ret.append(batch_original_labels)

            yield ret
			
def predicting(images, image_path, labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax')):

	labels_format = {'class_id': labels_output_format.index('class_id'),
						  'xmin': labels_output_format.index('xmin'),
						  'ymin': labels_output_format.index('ymin'),
						  'xmax': labels_output_format.index('xmax'),
						  'ymax': labels_output_format.index('ymax')}



	convert_to_3_channels = ConvertTo3Channels()
	resize = Resize(height=img_height, width=img_width)

	generate_pre = Data_Generator(image=images, image_path=image_path, labels_format=labels_format)
	predict_generator = generate_pre.generate(
								 batch_size=1,
								 transformations=[convert_to_3_channels, resize],
								 label_encoder=None,
								 returns={'processed_images',
										  'filenames',
										  'inverse_transform',
										  'original_images',
										  'original_labels'},
								 keep_images_without_gt=False)


	batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)

	i = 0

	print("Image:", "????")
	
	global graph
	with graph.as_default():
		y_pred = model.predict(batch_images)

	y_pred_decoded = decode_detections(y_pred,
									   confidence_thresh=0.25,
									   iou_threshold=0.4,
									   top_k=200,
									   normalize_coords=normalize_coords,
									   img_height=img_height,
									   img_width=img_width)

	print(y_pred_decoded)

	y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)
	# print(y_pred_decoded_inv)
	np.set_printoptions(precision=2, suppress=True, linewidth=90)
	# print("Predicted boxes:\n")
	# print('   class   conf xmin   ymin   xmax   ymax')
	# print(y_pred_decoded_inv[i])

	# Set the colors for the bounding boxes
	# plt.figure(figsize=(20, 12))
	# plt.imshow(batch_original_images[i])
	# colors = plt.cm.hsv(np.linspace(0, 1, n_classes + 1)).tolist()
	# current_axis = plt.gca()
	#
	# for box in y_pred_decoded_inv[i]:
	#     xmin = box[2]
	#     ymin = box[3]
	#     xmax = box[4]
	#     ymax = box[5]
	#     # color = colors[int(box[0])]
	#     # plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2)
	#     color = colors[int(box[0])]
	#     label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
	#     current_axis.add_patch(
	#         plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
	#     current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})
	#
	# # plt.show()
	# plt.xticks([])
	# plt.yticks([])
	# plt.savefig(self.image_path)

	# K.clear_session()
	# gc.collect()

	return y_pred_decoded_inv


