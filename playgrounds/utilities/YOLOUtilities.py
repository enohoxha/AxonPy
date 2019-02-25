import numpy as np

from keras.layers import Permute, Reshape, Lambda, add
from keras.layers import Conv2D, BatchNormalization, LeakyReLU
from keras import regularizers, initializers
import tensorflow as tf


def logistic(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_out = np.exp(x - np.max(x, axis=-1)[..., None])
    return exp_out / np.sum(exp_out, axis=-1)[..., None]


# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression(boxes, labels, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes	
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        i = idxs[-1]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:-1]])
        yy1 = np.maximum(y1[i], y1[idxs[:-1]])
        xx2 = np.minimum(x2[i], x2[idxs[:-1]])
        yy2 = np.minimum(y2[i], y2[idxs[:-1]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:-1]]

        # delete all indexes from the index list that have
        idxs = (idxs[:-1])[overlap < overlapThresh]

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick], labels[pick]


def getBoundingBoxesFromNetOutput(clf, anchors, confidence_threshold, cell_size):
    pw, ph = anchors[:, 0], anchors[:, 1]
    cell_inds = np.arange(clf.shape[1])

    tx = clf[..., 0]
    ty = clf[..., 1]
    tw = clf[..., 2]
    th = clf[..., 3]
    to = clf[..., 4] # object coefidence

    sftmx = softmax(clf[..., 5:])
    predicted_labels = np.argmax(sftmx, axis=-1)
    class_confidences = np.max(sftmx, axis=-1)


    bx = logistic(tx) + cell_inds[None, :, None]
    by = logistic(ty) + cell_inds[:, None, None]
    bw = pw * np.exp(tw) / 2
    bh = ph * np.exp(th) / 2
    object_confidences = logistic(to)

    left = bx - bw
    right = bx + bw
    top = by - bh
    bottom = by + bh

    boxes = np.stack((
        left, top, right, bottom
    ), axis=-1) * cell_size
    final_confidence = class_confidences * object_confidences
    boxes = boxes[final_confidence > confidence_threshold].reshape(-1, 4).astype(np.int32)
    labels = predicted_labels[final_confidence > confidence_threshold]
    return boxes, labels


def yoloPostProcess(yolo_output, priors, maxsuppression=True, maxsuppressionthresh=0.5, classthresh=0.3, cell_size=32):
    allboxes = []
    for o in yolo_output:
        boxes, labels = getBoundingBoxesFromNetOutput(o, priors, confidence_threshold=classthresh, cell_size=cell_size)
        if maxsuppression and len(boxes) > 0:
            boxes, labels = non_max_suppression(boxes, labels, maxsuppressionthresh)
        allboxes.append((boxes, labels))

    return allboxes



def reorg(input_tensor, stride):
    _, h, w, c = input_tensor.get_shape().as_list()

    channel_first = Permute((3, 1, 2))(input_tensor)

    reshape_tensor = Reshape((c // (stride ** 2), h, stride, w, stride))(channel_first)
    permute_tensor = Permute((3, 5, 1, 2, 4))(reshape_tensor)
    target_tensor = Reshape((-1, h // stride, w // stride))(permute_tensor)

    channel_last = Permute((2, 3, 1))(target_tensor)
    return Reshape((h // stride, w // stride, -1))(channel_last)


def conv_batch_lrelu(input_tensor, numfilter, dim, strides=1):
    '''
    Create 3 layers conv + batch + relu
    :param input_tensor: prev model
    :param numfilter: number of filters for conv
    :param dim: dimension used in conv layer
    :param strides: strides on conv layer
    :return: keras model api LeakyReLU
    '''
    input_tensor = Conv2D(numfilter, (dim, dim), strides=strides, padding='same',
                          kernel_regularizer=regularizers.l2(0.0005),
                          kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                          use_bias=False
                          )(input_tensor)
    input_tensor = BatchNormalization()(input_tensor)
    return LeakyReLU(alpha=0.1)(input_tensor)


def NetworkInNetwork(input_tensor, dims):
    for d in dims:
        input_tensor = conv_batch_lrelu(input_tensor, *d)
    return input_tensor


def Upsample(stride):
    def _upsample(x):
        h, w = x.get_shape().as_list()[1:3]
        return tf.image.resize_images(x, [h * 2, w * 2], align_corners=True)

    return Lambda(_upsample)


def residual(input_tensor, num_blocks, dims):
    out = input_tensor
    for _ in range(num_blocks):
        out = add([NetworkInNetwork(out, dims), out])
    return out
