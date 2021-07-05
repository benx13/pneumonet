
import tempfile
import time

import numpy as np
import streamlit as st
import cv2
import torch
from src.preprocessing.enhancment.cv2_enhance import enhance
from torchvision.transforms.functional import to_pil_image

from torchcam.cams import ScoreCAM
from torchcam.utils import overlay_mask
from src.utils.utils import f_masked
import cv2
from tqdm import tqdm
import time

import numpy as np
from PIL import Image
from torchvision import models, transforms

import torch

from torchvision.transforms import Compose
import os
import torch.nn as nn
from src.utils.utils import masked as ms
from imutils import paths

import matplotlib.pyplot as plt

from src.preprocessing.bonsup.resnetbs import resnet_bs

from src.preprocessing.segmentation.VAE import uVAE
from src.utils.utils import saveMask, loadDCM

from src.preprocessing.bonsup.resnetbs import resnet_bs
from src.preprocessing.segmentation.VAE import uVAE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



if 'resnet18masked' not in st.session_state:
    st.session_state.resnet18masked = models.resnet18(pretrained=False)

    st.session_state.resnet18masked.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 4))
    st.session_state.resnet18masked.to(device)
    st.session_state.resnet18masked.load_state_dict(torch.load('pretrained_models/xresnet18masked.pt', map_location=torch.device('cpu')))

def getLabel(label):
    dic = {0:'Normal', 1:'Covid', 2:'Lung Opacity', 3:'Pneumonia'}
    return dic[label]

def plot_hm(image):
    cam_extractor = ScoreCAM(st.session_state.resnet18masked)
    y_test_pred = st.session_state.resnet18masked(image.unsqueeze(0).to(device))
    _, y_pred_tags = torch.max(y_test_pred, dim = 1)
    activation_map = cam_extractor(y_test_pred.squeeze(0).argmax().item(), y_test_pred)
    return getLabel(y_pred_tags.item()), overlay_mask(to_pil_image(image), to_pil_image(activation_map, mode='F'), alpha=0.5)

if 'VAE'  not in st.session_state:
    st.session_state.VAE = uVAE(nhid= 16, nlatent=8)
    st.session_state.VAE.load_state_dict(torch.load('pretrained_models/VAE.pt', map_location=torch.device("cpu")))
    st.session_state.VAE.to(device)
    st.session_state.VAE.eval()
if 'resnetbs' not in st.session_state:
    st.session_state.resnetbs = resnet_bs(num_filters=64, num_res_blocks=16, res_block_scaling=0.1)
    st.session_state.resnetbs.load_weights('pretrained_models/resnet-bs.h5')



def f(image):
    # result = add(1, 2)
    # st.write('result: %s' % result)
    #st.title('3-Select a preprocessing technique')

    x = cv2.resize(image, (256, 256))
    x = x.astype('float32') / 255

    x1 = np.expand_dims(x, axis=0)
    t = time.time()
    bs = st.session_state.resnetbs.predict(x1)
    bs = np.reshape(bs, (256, 256))
    bs = bs * 255
    img, roi, h, w, hLoc, wLoc, imH, imW = loadDCM(bs, preprocess=False)

    img = img.to(device)
    _, pred = st.session_state.VAE(img)
    pred = torch.sigmoid(pred.cpu() * roi)
    pred = saveMask(pred.squeeze(), h, w, hLoc, wLoc, imH, imW)
    pred = torch.from_numpy(pred)
    pred = torch.unsqueeze(pred, 0)
    pred = torch.unsqueeze(pred, 1)
    pred = pred.squeeze().numpy()

    pred = pred[65:510]
    pred = 1.0 * (pred > 0.5)

    img = img.cpu().squeeze()[65:510]
    img = cv2.resize(np.array(img), (256, 256))

    pred = cv2.resize(np.array(pred), (256, 256))
    print(img)
    print(pred)

    printmaskbs= f_masked(img, pred, pred)
    printmaskx= f_masked(x, pred, pred)

    #pred = bs()
    bsmasked = img * pred * 255
    bsmasked = cv2.resize(np.array(bsmasked), (256, 256))


    masked = x * pred * 255
    masked = cv2.resize(np.array(masked), (256, 256))

    x = x * 255
    t = Compose([transforms.ToPILImage(),
                 transforms.Resize((224, 224),
                                   interpolation=Image.NEAREST),
                 transforms.ToTensor()])
    masked = masked.astype('uint8')
    bs = bs.astype('uint8')
    x = x.astype('uint8')
    bsmasked = bsmasked.astype('uint8')
    return (bs, masked, x, bsmasked, enhance(x), printmaskbs, printmaskx)



st.title('PneumoNet')
st.title("1- Upload an xray image or use default")
#st.write("2- Pick pick a preprocessing techinque")
#st.write("3- Predicting the image")
#st.sidebar.title("About")

#st.sidebar.info("This is a demo application written to help you understand Streamlit. The application identifies the animal in the picture. It was built using a Convolution Neural Network (CNN).")
#st.title('Select a model')
#modelname = st.sidebar.selectbox('Select model', ['resnet'])

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
if (file is not None):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    name = tfile.name
    st.write('please wait reading image ...')

else:
    name = 'images/chexnet.png'
    st.write('please wait reading default image ...')

image = cv2.imread(name, 0)
if 'prepi' not in st.session_state:
    st.session_state.prepi = f(image)
st.image(image, use_column_width=True)
st.title('2-Pick a preporcessing technique')
#bs, masked, x, bsmasked = f(image)

t = Compose([transforms.ToPILImage(),
                 transforms.Resize((224, 224),
                                   interpolation=Image.NEAREST),
                 transforms.ToTensor()])
if st.button('No preprocessing'):
    st.title('3-Classifying image')
    img = np.stack((np.array(st.session_state.prepi[2]),) * 3, axis=-1)
    #st.write("applying mask to image")
    #st.image(st.session_state.prepi[4], caption="Masked on image", use_column_width=True)
    #st.write("generating preprocessed image")
    #st.image(img, caption="Preporcessed image", use_column_width=True)
    st.write('classifying image  ...')
    y_tag, hm = plot_hm(t(img))
    st.title("Image diagnosed as "+ y_tag)
    st.title('4-Generating heatmap')
    st.image(hm, caption="The HeatMap", use_column_width=True)

if st.button('Bone Suppression'):
    st.title('3-Predicting Image Class ')
    img = np.stack((np.array(st.session_state.prepi[0]),) * 3, axis=-1)
    #st.write("applying mask to image")
    #st.image(st.session_state.prepi[4], caption="Masked on image", use_column_width=True)
    st.write("generating preprocessed image")
    st.image(img, caption="Preporcessed image", use_column_width=True)
    st.write('classifying image  ...')
    y_tag, hm = plot_hm(t(img))
    st.title("Image diagnosed as "+ y_tag)
    st.title('4-Generating heatmap')
    st.image(hm, caption="The HeatMap", use_column_width=True)

if st.button('Masked'):
    st.title('3-Predicting Image Class ')
    img = np.stack((np.array(st.session_state.prepi[1]),) * 3, axis=-1)
    st.write("applying mask to image")
    st.image(st.session_state.prepi[6], caption="Masked on image", use_column_width=True, clamp=True, channels='BGR')
    st.write("generating preprocessed image")
    st.image(img, caption="Preporcessed image", use_column_width=True)
    st.write('classifying image  ...')
    y_tag, hm = plot_hm(t(img))
    st.title("Image diagnosed as "+ y_tag)
    st.title('4-Generating heatmap')
    st.image(hm, caption="The HeatMap", use_column_width=True)

if st.button('Bone Suppression Masked'):
    st.title('3-Predicting Image Class ')
    img = np.stack((np.array(st.session_state.prepi[3]),) * 3, axis=-1)
    st.write("applying mask to image")
    st.image(st.session_state.prepi[5], caption="Masked on image", use_column_width=True , clamp=True, channels='BGR')
    st.write("generating preprocessed image")
    st.image(img, caption="Preporcessed image", use_column_width=True)
    st.write('classifying image  ...')
    y_tag, hm = plot_hm(t(img))
    st.title("Image diagnosed as "+ y_tag)
    st.title('4-Generating heatmap')
    st.image(hm, caption="The HeatMap", use_column_width=True)

if st.button('Enhance'):
    st.title('3-Predicting Image Class')
    img = np.stack((np.array(st.session_state.prepi[4]),) * 3, axis=-1)
    st.write("generating preprocessed image")
    st.image(img, caption="Preporcessed image", use_column_width=True)
    st.write('classifying image  ...')
    y_tag, hm = plot_hm(t(img))
    st.title("Image diagnosed as " + y_tag)
    st.title('4-Generating heatmap')
    st.image(hm, caption="The HeatMap", use_column_width=True)
