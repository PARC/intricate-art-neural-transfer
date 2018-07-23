from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from scipy.misc import imread, imresize, imsave, fromimage, toimage
from scipy.optimize import fmin_l_bfgs_b
import numpy as np 
import time
import argparse
import warnings
import os
from PIL import Image
import PIL.ImageOps
import tensorflow as tf

from tensorflow.python.keras._impl.keras.models import Model
from tensorflow.python.keras._impl.keras.engine import Input
from tensorflow.python.keras._impl.keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras.utils.data_utils import get_file
from tensorflow.python.keras._impl.keras.utils.layer_utils import convert_all_kernels_in_model
from tensorflow.python.keras._impl.keras.applications.vgg16 import VGG16


TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

TF_19_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')

parser.add_argument('style_image_paths', metavar='ref', nargs='+', type=str,
                    help='Path to the style reference image.')

parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')

parser.add_argument("--image_size", dest="img_size", default=600, type=int,
                    help='Minimum image size')

parser.add_argument("--content_weight", dest="content_weight", default=0.025, type=float,
                    help="Weight of content")

parser.add_argument("--style_weight", dest="style_weight", nargs='+', default=[1], type=float,
                    help="Weight of style, can be multiple for multiple styles")

parser.add_argument("--style_scale", dest="style_scale", default=1.0, type=float,
                    help="Scale the weighing of the style")

parser.add_argument("--total_variation_weight", dest="tv_weight", default=8.5e-5, type=float,
                    help="Total Variation weight")

parser.add_argument("--num_iter", dest="num_iter", default=10, type=int,
                    help="Number of iterations")

parser.add_argument("--model", default="vgg16", type=str,
                    help="Choices are 'vgg16' and 'vgg19'")

parser.add_argument("--content_loss_type", default=0, type=int,
                    help='Can be one of 0, 1 or 2. Readme contains the required information of each mode.')

parser.add_argument("--rescale_image", dest="rescale_image", default="False", type=str,
                    help="Rescale image after execution to original dimentions")

parser.add_argument("--rescale_method", dest="rescale_method", default="bilinear", type=str,
                    help="Rescale image algorithm")

parser.add_argument("--maintain_aspect_ratio", dest="maintain_aspect_ratio", default="True", type=str,
                    help="Maintain aspect ratio of loaded images")

parser.add_argument("--content_layer", dest="content_layer", default="block5_conv2", type=str,
                    help="Content layer used for content loss.")

parser.add_argument("--init_image", dest="init_image", default="content", type=str,
                    help="Initial image used to generate the final image. Options are 'content', 'noise', or 'gray'")

parser.add_argument("--pool_type", dest="pool", default="max", type=str,
                    help='Pooling type. Can be "ave" for average pooling or "max" for max pooling')

parser.add_argument('--background_color', dest="bg_color", default='#ffffff', help='Specify background color')

parser.add_argument('--background_image', dest='bg_image', default=None, help="Specify background image")

parser.add_argument('--min_improvement', default=0.0, type=float,
                    help='Defines minimum improvement required to continue script')


def str_to_bool(v):
    return v.lower() in ("true", "yes", "t", "1")

''' Arguments '''

args = parser.parse_args()
base_image_path = args.base_image_path
style_reference_image_paths = args.style_image_paths
result_prefix = args.result_prefix


silhouette = Image.open(base_image_path).convert('L')
inverted_silhouette = PIL.ImageOps.invert(silhouette)


style_image_paths = []
for style_image_path in style_reference_image_paths:
    style_image_paths.append(style_image_path)


rescale_image = str_to_bool(args.rescale_image)
maintain_aspect_ratio = str_to_bool(args.maintain_aspect_ratio)


def hex_to_rgb(h):
    hh = h.lstrip("#")
    rgb = tuple(int(hh[i:i+2], 16) for i in (0, 2 ,4))
    return rgb

bg_color = hex_to_rgb(args.bg_color)

bg_image = args.bg_image

# these are the weights of the different loss components
content_weight = args.content_weight
total_variation_weight = args.tv_weight

style_weights = []

if len(style_image_paths) != len(args.style_weight):
    print("Mismatch in number of style images provided and number of style weights provided. \n"
          "Found %d style images and %d style weights. \n"
          "Equally distributing weights to all other styles." % (len(style_image_paths), len(args.style_weight)))

    weight_sum = sum(args.style_weight) * args.style_scale
    count = len(style_image_paths)

    for i in range(len(style_image_paths)):
        style_weights.append(weight_sum / count)
else:
    for style_weight in args.style_weight:
        style_weights.append(style_weight * args.style_scale)

# Decide pooling function
pooltype = str(args.pool).lower()
assert pooltype in ["ave", "max"], 'Pooling argument is wrong. Needs to be either "ave" or "max".'

pooltype = 1 if pooltype == "ave" else 0

read_mode = "gray" if args.init_image == "gray" else "color"

# dimensions of the generated picture.
img_width = img_height = 0

img_WIDTH = img_HEIGHT = 0
aspect_ratio = 0

assert args.content_loss_type in [0, 1, 2], "Content Loss Type must be one of 0, 1 or 2"

# util function to open, resize and format pictures into appropriate tensors
def preprocess_image(image_path, load_dims=False, read_mode="color"):
    global img_width, img_height, img_WIDTH, img_HEIGHT, aspect_ratio

    mode = "RGB" if read_mode == "color" else "L"
    img = imread(image_path, mode=mode)  # Prevents crashes due to PNG images (ARGB)

    if mode == "L":
        # Expand the 1 channel grayscale to 3 channel grayscale image
        temp = np.zeros(img.shape + (3,), dtype=np.uint8)
        temp[:, :, 0] = img
        temp[:, :, 1] = img.copy()
        temp[:, :, 2] = img.copy()

        img = temp

    if load_dims:
        img_WIDTH = img.shape[0]
        img_HEIGHT = img.shape[1]
        aspect_ratio = float(img_HEIGHT) / img_WIDTH

        img_width = args.img_size
        if maintain_aspect_ratio:
            img_height = int(img_width * aspect_ratio)
        else:
            img_height = args.img_size

    img = imresize(img, (img_width, img_height)).astype('float32')

    # RGB -> BGR
    img = img[:, :, ::-1]

    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68

    img = np.expand_dims(img, axis=0)
    return img


# util function to convert a tensor into a valid image
def deprocess_image(x):
    x = x.reshape((img_width, img_height, 3))

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # BGR -> RGB
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


def load_mask_sil(invert_sil, shape):
    width, height, _ = shape
    invert_array = np.array(invert_sil.convert('L'))
    mask = imresize(invert_sil, (width, height), interp='bicubic').astype('float32')

    # Perform binarization of mask
    mask[mask <= 127] = 0
    mask[mask > 128] = 255

    max = np.amax(mask)
    mask /= max

    return mask

   

# util function to apply mask to generated image
def mask_content(content_path, generated, mask, bg_color=bg_color):
    content_image = imread(content_path, mode='RGB')
    content_image = imresize(content_image, (img_width, img_height), interp='bicubic')
    width, height, channels = generated.shape
    if bg_image is not None:
        background_image = imread(bg_image, mode='RGB')
        background_image = imresize(background_image, (img_width, img_height), interp='bicubic')
        for i in range(width):
            for j in range(height):
                if mask[i,j] == 0:
                    generated[i, j, :] = background_image[i, j, :]
    else:
        for i in range(width):
            for j in range(height):
                if mask[i, j] == 0.:
                    for k in range(3):
                        generated[i,j][k] = bg_color[k]

    return generated



def pooling_func(x):
    if pooltype == 1:
        return AveragePooling2D((2, 2), strides=(2, 2))(x)
    else:
        return MaxPooling2D((2, 2), strides=(2, 2))(x)


# get tensor representations of our images
base_image = K.variable(preprocess_image(base_image_path, True, read_mode=read_mode))

style_reference_images = []
for style_path in style_image_paths:
    style_reference_images.append(K.variable(preprocess_image(style_path)))

# this will contain our generated image
combination_image = K.placeholder((1, img_width, img_height, 3))

image_tensors = [base_image]
for style_image_tensor in style_reference_images:
    image_tensors.append(style_image_tensor)
image_tensors.append(combination_image)

nb_tensors = len(image_tensors)
print("nb_tensors", nb_tensors)
nb_style_images = nb_tensors - 2 # Content and Output image not considered

# combine the various images into a single Keras tensor
input_tensor = K.concatenate(image_tensors, axis=0)

shape = (nb_tensors, img_width, img_height, 3)
inp_shape = (img_width, img_height, 3)

ip = Input(tensor=input_tensor, batch_shape=shape)

model = VGG16(include_top=False, weights='imagenet', input_tensor = input_tensor,
              input_shape=inp_shape, pooling='max')

print('Model loaded.')

# get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
shape_dict = dict([(layer.name, layer.output_shape) for layer in model.layers])

# compute the neural style loss
# first we need to define 4 util functions

# Improvement 1
# the gram matrix of an image tensor (feature-wise outer product) using shifte, return_mask_img = Trued activations
def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features - 1, K.transpose(features - 1))
    return gram


# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image

def style_loss(style, combination, nb_channels=None):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3

    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_width * img_height
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image
def content_loss(base, combination):
    channel_dim = -1

    try:
        channels = K.int_shape(base)[channel_dim]
    except TypeError:
        channels = K.shape(base)[channel_dim]
    size = img_width * img_height

    if args.content_loss_type == 1:
        multiplier = 1. / (2. * (channels ** 0.5) * (size ** 0.5))
    elif args.content_loss_type == 2:
        multiplier = 1. / (channels * size)
    else:
        multiplier = 1.

    return multiplier * K.sum(K.square(combination - base))


# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent
def total_variation_loss(x):
    assert K.ndim(x) == 4
    a = K.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, 1:, :img_height - 1, :])
    b = K.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, :img_width - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

if args.model == "vgg19":
    feature_layers = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2',
                  'block3_conv3', 'block3_conv4', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4',
                  'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4']
else:

    feature_layers = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2',
                       'block3_conv3', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block5_conv1', 'block5_conv2',
                       'block5_conv3']

# combine these loss functions into a single scalar
loss = K.variable(0.)
layer_features = outputs_dict[args.content_layer]
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[nb_tensors - 1, :, :, :]
loss = loss + content_weight * content_loss(base_image_features,
                                      combination_features)
# Improvement 2
# Use all layers for style feature extraction and reconstruction
nb_layers = len(feature_layers) - 1

channel_index =  -1

# Improvement 3 : Chained Inference without blurring
for i in range(len(feature_layers) - 1):
    layer_features = outputs_dict[feature_layers[i]]
    shape = shape_dict[feature_layers[i]]
    combination_features = layer_features[nb_tensors - 1, :, :, :]
    style_reference_features = layer_features[1:nb_tensors - 1, :, :, :]
    sl1 = []
    for j in range(nb_style_images):
        sl1.append(style_loss(style_reference_features[j], combination_features,  shape))

    layer_features = outputs_dict[feature_layers[i + 1]]
    shape = shape_dict[feature_layers[i + 1]]
    combination_features = layer_features[nb_tensors - 1, :, :, :]
    style_reference_features = layer_features[1:nb_tensors - 1, :, :, :]
    sl2 = []
    for j in range(nb_style_images):
        sl2.append(style_loss(style_reference_features[j], combination_features, shape))
        

    for j in range(nb_style_images):
        sl = sl1[j] - sl2[j]

        # Improvement 4
        # Geometric weighted scaling of style loss
        loss += (style_weights[j] / (2 ** (nb_layers - (i + 1)))) * sl

loss += total_variation_weight * total_variation_loss(combination_image)

# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_image)

outputs = [loss]
if type(grads) in {list, tuple}:
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)


def eval_loss_and_grads(x):
    x = x.reshape((1, img_width, img_height, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()

# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss


if "content" in args.init_image or "gray" in args.init_image:
    x = preprocess_image(base_image_path, True, read_mode=read_mode)
elif "noise" in args.init_image:
    x = np.random.uniform(0, 255, (1, img_width, img_height, 3)) - 128.

else:
    print("Using initial image : ", args.init_image)
    x = preprocess_image(args.init_image, read_mode=read_mode)


num_iter = args.num_iter
prev_min_val = -1

improvement_threshold = float(args.min_improvement)

for i in range(num_iter):
    print("Starting iteration %d of %d" % ((i + 1), num_iter))
    start_time = time.time()

    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)

    if prev_min_val == -1:
        prev_min_val = min_val

    improvement = (prev_min_val - min_val) / prev_min_val * 100

    print("Current loss value:", min_val, " Improvement : %0.3f" % improvement, "%")
    prev_min_val = min_val
    # save current generated image
    img = deprocess_image(x.copy())

    if not rescale_image:
        img_ht = int(img_width * aspect_ratio)
        print("Rescaling Image to (%d, %d)" % (img_width, img_ht))
        img = imresize(img, (img_width, img_ht), interp=args.rescale_method)

    if rescale_image:
        print("Rescaling Image to (%d, %d)" % (img_WIDTH, img_HEIGHT))
        img = imresize(img, (img_WIDTH, img_HEIGHT), interp=args.rescale_method)
    
    if i == num_iter-1:
        fname = result_prefix + "_pattern_output.png"

        mask = load_mask_sil(inverted_silhouette, img.shape)
        final_img = mask_content(base_image_path, img, mask)
        end_time = time.time()
        imsave(fname, final_img)
        print("Image saved as", fname)
        print("Iteration %d completed in %ds" % (i + 1, end_time - start_time))

    if improvement_threshold is not 0.0:
        if improvement < improvement_threshold and improvement is not 0.0:
            print("Improvement (%f) is less than improvement threshold (%f). Early stopping script." %
                  (improvement, improvement_threshold))
            exit()
