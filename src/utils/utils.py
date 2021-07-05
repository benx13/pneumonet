from collections import OrderedDict

import numpy as np
import torch
from pydicom import dcmread
from skimage import morphology, color
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist as equalize
from skimage.io import imread
from skimage.measure import label
from skimage.morphology import binary_closing as closing
from skimage.morphology import disk
from skimage.transform import resize

torch.manual_seed(42)
np.random.seed(42)


def get_mean_std(loader):
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean, std


def normalize(
        min_old,
        max_old,
        min_new,
        max_new,
        val,
):
    """Normalizes values to the interval [min_new, max_new]

....Parameters:
........min_old: min value from old base.
........max_old: max value from old base.
........min_new: min value from new base.
........max_new: max value from new base.
........val: float or array-like value to be normalized.
...."""

    ratio = (val - min_old) / (max_old - min_old)
    normalized = (max_new - min_new) * ratio + min_new
    return normalized.astype(np.uint8)


def histogram(data):
    """Generates the histogram for the given data.

....Parameters:
........data: data to make the histogram.

....Returns: histogram, bins.
...."""

    (pixels, count) = np.unique(data, return_counts=True)
    hist = OrderedDict()

    for i in range(len(pixels)):
        hist[pixels[i]] = count[i]

    return np.array(list(hist.values())), np.array(list(hist.keys()))


def to_grayscale(image):
    red_v = image[:, :, 0] * 0.299
    green_v = image[:, :, 1] * 0.587
    blue_v = image[:, :, 2] * 0.144
    image = red_v + green_v + blue_v

    return image.astype(np.uint8)


def clip_histogram(hist, bins, clip_limit):
    """Clips the given histogram.

....Parameters:
........hist: frequencies of each pixel.
........bins: pixels.
........clip_limit: limit to pixel frequencies.

....Returns the clipped hist.
...."""

    n_bins = len(bins)

    # Removing values above clip_limit

    excess = 0
    for i in range(n_bins):
        if hist[i] > clip_limit:
            excess += hist[i] - clip_limit
            hist[i] = clip_limit

    # # Redistributing exceeding values ##
    # Calculating the values to be put on all bins

    for_each_bin = excess // n_bins

    # Calculating the values left

    leftover = excess % n_bins

    hist += for_each_bin
    for i in range(leftover):
        hist[i] += 1

    return hist


def calculate_cdf(hist, bins):
    """Calculates the normalized CDF (Cumulative Distribution Function)
....for the histogram.

....Parameters:
........hist: frequencies of each pixel.
........bins: pixels.

....Returns the CDF in a dictionary.
...."""

    # Calculating probability for each pixel

    pixel_probability = hist / hist.sum()

    # Calculating the CDF (Cumulative Distribution Function)

    cdf = np.cumsum(pixel_probability)

    cdf_normalized = cdf * 255

    hist_eq = {}
    for i in range(len(cdf)):
        hist_eq[bins[i]] = int(cdf_normalized[i])

    return hist_eq


def loadDCM(dcm, preprocess=True, dicom=False, p=1):
    wLoc = 448
    ### Load input dico
    dcm = dcm / dcm.max()

    if preprocess:
        dcm = equalize(dcm)
        # img = util.invert(img)
        # img = preprocessing.enhancement.hfe(img)

    print(dcm.shape)
    if len(dcm.shape) > 2:
        dcm = rgb2gray(dcm[:, :, :3])
    hLoc = int((dcm.shape[0] / (dcm.shape[1] / wLoc)))
    if hLoc > 576:
        hLoc = 576
        wLoc = int((dcm.shape[1] / (dcm.shape[0] / hLoc)))
    img = resize(dcm, (448, 510))
    img = torch.Tensor(img)
    pImg = torch.zeros((640, 512))
    h = (int((576 - hLoc) / 2)) + p
    w = int((448 - wLoc) / 2) + p
    roi = torch.zeros(pImg.shape)
    if w == p:
        pImg[np.abs(h):(h + img.shape[0]), p:-p] = img
        roi[np.abs(h):(h + img.shape[0]), p:-p] = 1.0
    else:
        pImg[p:-p, np.abs(w):(w + img.shape[1])] = img
        roi[p:-p, np.abs(w):(w + img.shape[1])] = 1.0
    imH = dcm.shape[0]
    imW = dcm.shape[1]
    pImg = pImg.unsqueeze(0).unsqueeze(0)
    return pImg, roi, h, w, hLoc, wLoc, imH, imW


def largestCC(lImg, num=2):
    cIdx = np.zeros(num, dtype=int)
    count = np.bincount(lImg.flat)
    count[0] = 0  # Mark background count to zero
    lcc = np.zeros(lImg.shape, dtype=bool)
    if len(count) == 2:
        num = 1
    for i in range(num):
        cIdx[i] = np.argmax(count)
        count[cIdx[i]] = 0
        lcc += (lImg == cIdx[i])

    return lcc


def postProcess(img, s=11):
    bImg = (img > 0.5)
    if len(bImg.shape) > 2:
        bImg = bImg[:, :, -1]
    sEl = disk(s)
    lImg = label(bImg)
    lcc = largestCC(lImg)  # Obtain the two largest connected components
    pImg = closing(lcc, sEl)
    return pImg.astype(float)


def saveMask(img, h, w, hLoc, wLoc, imH, imgW, no_post=False):
    p = 32
    img = img.data.numpy()
    imgIp = img.copy()

    if w == p:
        img = resize(img[np.abs(h):(h + hLoc), p:-p],
                     (imH, imgW), preserve_range=True)
    else:
        img = resize(img[p:-p, np.abs(w):(w + wLoc)],
                     (imH, imgW), preserve_range=True)
    return postProcess(imgIp)


def extract_bboxes(m):
    horizontal_indicies = np.where(np.any(m, axis=0))[0]
    vertical_indicies = np.where(np.any(m, axis=1))[0]
    x1, x2 = horizontal_indicies[[0, -1]]
    y1, y2 = vertical_indicies[[0, -1]]
    # x2 and y2 should not be part of the box. Increment by 1.
    x2 += 1
    y2 += 1
    boxe = np.array([y1, x1, y2, x2])
    return boxe.astype(np.int32)


def masked(img, gt, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""

    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    boundary = morphology.dilation(gt, morphology.disk(3)) - gt
    color_mask[mask == 1] = [0, 1, 1]
    color_mask[boundary == 1] = [1, 0, 0]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked


def f_masked(img, gt, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""

    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    boundary = morphology.dilation(gt, morphology.disk(3)) - gt

    color_mask[mask == 1] = [0, 1, 1]
    color_mask[boundary == 1] = [1, 0, 0]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
    img_masked = color.hsv2rgb(img_hsv)
    return img_masked