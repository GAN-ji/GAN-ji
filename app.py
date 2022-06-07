import os
import sys
import torch

from utils import *
from model import *

import streamlit as st
from streamlit.script_run_context import SCRIPT_RUN_CONTEXT_ATTR_NAME
from PIL import Image

from contextlib import contextmanager
from threading import current_thread
from io import StringIO


@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b + '')
                output_func(buffer.getvalue() + '')
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write

@contextmanager
def st_stdout(dst):
    "this will show the prints"
    st_redirect(sys.stdout, dst)
    #yield

@contextmanager
def st_stderr(dst):
    "This will show the logging"
    with st_redirect(sys.stderr, dst):
        yield


# def update_latent():
#     global latent
#     latent = torch.unsqueeze(samples_z[row_number-1][col_number-1], dim=0)
#     latent = g_ema.get_latent(latent)

if __name__ == '__main__':
    
    st.title("GAN-ji🙂")
    st.markdown("")

    # https://shields.io/
    # st.markdown("<br>", unsafe_allow_html=True)

        
    st.info('''
    Aiffel YJ GAN-ji Team 
    ''')

    col1, col2 = st.columns([2,2])
    open_notion = col1.button("🚀 Open in notion")  # logic handled further down
    open_github = col2.button("💻 Github")  # logic handled further down

    
    if open_notion:
        utils.open_link('https://bottlenose-client-925.notion.site/GAN-ji-0f10fdc083a244f5b15a97dd1c797d71')
    if open_github:
        utils.open_link('https://github.com/GAN-ji')


    # download model
    device = 'cuda'

    #latent = 512
    n_mlp = 8
    size = 256
    channel_multiplier = 2
    ckpt = 'networks/635000.pt' 

    g_ema = Generator(
        size, 512, n_mlp, channel_multiplier=channel_multiplier
    ).to(device)
    checkpoint = torch.load(ckpt) 

    g_ema.load_state_dict(checkpoint["g_ema"])

    mean_latent = None

    eigvec = torch.load('factor.pt')["eigvec"].to('cuda')

    trunc = g_ema.mean_latent(4096)
    samples_z = []
    # global latent 
    latent = None


   # --------------------------------

    st.markdown('---')
    st.header('Generate Random Images!')
    

    col1,col2,col3 = st.columns([3,2,2])
    with col2:
        button1 = st.button("start GAN-ji😎")

    if button1:
        samples_z = generate(1, g_ema, device, mean_latent)
        st.session_state['samples_z'] = samples_z
    
    if 'samples_z' in st.session_state:
        image = Image.open('sample/samples.png')
        st.image(image, caption='Random GANjis')
        st.success('welcome to GAN-ji world!')

        col1,col2,col3 = st.columns([3,2,2])
        with col2:
            with open("sample/samples.png", "rb") as file:
                btn = st.download_button(
                    label="Download Grid",
                    data=file,
                    file_name="GAN-ji_Grid.png",
                    mime="image/png"
                )

    # --------------------------------

    st.markdown('---')
    st.header('Customize your Emoji!')
    device = 'cuda'

    st.info('''
    행 X 렬
    ''')
    col1, col2 = st.columns([1,1])
    with col1:
        row_number = st.number_input('row', min_value=1, max_value=4, value=1, step=1)
    with col2:
        col_number = st.number_input('col', min_value=1, max_value=5, value=1, step=1)
        
    #st.write(st.session_state.samples_z)
    if 'samples_z' in st.session_state:
        #st.write(st.session_state.samples_z.shape)
        st.session_state['latent'] = torch.unsqueeze(st.session_state['samples_z'][row_number-1][col_number-1], dim=0)
        st.session_state['latent'] = g_ema.get_latent(st.session_state['latent'])
        img, _ = g_ema(
        [st.session_state.latent],
        #truncation=0.7,
        truncation_latent=trunc,
        input_is_latent=True,
        )

        utils.save_image(
            img,
            f"output.png",
            normalize=True,
            range=(-1, 1),
            nrow=1,
        )
        #st.write(st.session_state.latent)

    col1, col2 = st.columns([2,2])
    with col1:  
        Feature1 = st.slider('expression', -10, 10, step=1, value=0)#, on_change=sefa1)
        Feature2 = st.slider('color', -10, 10, step=1, value=0)#, on_change=sefa2)
        Feature3 = st.slider('style', -10, 10, step=1, value=0)#, on_change=sefa3)
        Feature4 = st.slider('???', -10, 10, step=1, value=0)#, on_change=sefa4)
    features = [Feature1, Feature2, Feature3, Feature4]

    if sum(features)!=0 and 'samples_z' in st.session_state:
        #st.write(eigvec.shape)
        idx_list = [0,1,2,3]
        for i in range(len(idx_list)): 
            #st.write(type(latent))
            st.session_state['latent'] = sefa(idx_list[i], features[i], st.session_state['latent'], eigvec, trunc, g_ema)

    with col2:
        if 'latent' in st.session_state:
            image = Image.open('output.png')
            st.image(image, caption='Input Image', use_column_width=True)


    col1,col2,col3 = st.columns([3,2,2])
    with col2:
        with open("output.png", "rb") as file:
            btn = st.download_button(
                label="Download GAN-ji",
                data=file,
                file_name="GAN-ji_sample.png",
                mime="image/png"
            )
