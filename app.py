import streamlit as st
from xml.etree.ElementPath import prepare_descendant
import numpy as np
import os
import pandas as pd
from PIL import Image
import h5py
from pathlib import Path
import argparse

import torch
from torch.utils import data
from tqdm import tqdm

from data import IUSingleImage, MIMICSingeImage

import clip
from utils import nonpretrained_params

from run_test import predict, get_text_embeddings

st.title("Chest X-Ray Report Generation")
st.text("")
st.subheader(
    "This is a demo of the report generation model using CXR-RePaiR: Retrieval-Based Chest X-Ray Report Generation Using a Pre-trained Contrastive Language-Image Model")
image = Image.open('_assets/cxr-repair.png')

st.image(image, caption='CXR-RePaiR', use_column_width=True)

st.header("Dataset:")
st.text("In our case we used IU dataset because it is the only dataset that has both images and reports available."
        " \nHowever, the model can be trained on any dataset that has images and reports\navailable. So text "
        "embeddings are precalculated for IU dataset and same for the IU\ndataset images.")
st.text("We'll be using pretrained CLIP model checkpoint trained on MIMIC-CXR train set.\nIt is VIT-B/32 model."
        "We'll be using the same model for both image and text embeddings.")

# Prediction code
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, _ = clip.load("ViT-B/32", device=device, jit=False)
print("Loaded in pretrained model.")

clip_model_path = 'clip-imp-pretrained_128_6_after_4.pt'
model.load_state_dict(torch.load(clip_model_path, map_location=device))
model = model.to(device)

# precalculated corpus clip embeddings
corpus_embeddings_path = 'corpus_embeddings/' + 'clip_pretrained_mimic_train_sentence_embeddings.pt'
raw_impressions, text_embeddings = get_text_embeddings(corpus_embeddings_path, '')

st.header("Model Prediction:")

uploaded_file = st.file_uploader("Choose a CXR image file")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='CXR-IMG', use_column_width=True)

    with st.spinner('Wait for it...'):
        dset = IUSingleImage(uploaded_file, clip_pretrained=True)

        loader = torch.utils.data.DataLoader(dset, shuffle=False, batch_size=4)

        # select top report/sentences
        y_pred = predict(loader, text_embeddings, model, device, topk=2)
        report = raw_impressions[np.max(y_pred[0])]

    st.success('Done!')
    st.subheader("Generated Report:")
    st.info(report)
