import os
import sys
import torch

#import utils_
from model import *
from utils import *

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


if __name__ == '__main__':
    
    st.title("GAN-ji🙂")
    st.markdown("")

    # https://shields.io/
    # st.markdown("<br>", unsafe_allow_html=True)

        
    st.info('''
     "이모지는 영어보다도 의사소통 수단으로서 강력하다"  – 비비언 에반스 박사(언어, 디지털 통신 전문가)

    펜데믹 이후로 더욱 가속화된 디지털 시대에서 이모티콘은 거의 필수적이고, 사람들은 상황에 맞는 다양한 이모티콘을 이용하고 구매할 의사도 있는데 왜 아직까지 제대로 된 커스터마이징 이모지를 제공하는 서비스는 없을까? 라는 의문에서 저희 프로젝트가 시작되었습니다.
    친숙한 이모지들을 잠시나마 더 재밌게 사용해보는 경험이었으면 좋겠습니다. 
    ''')

    col1, col2 = st.columns([2,2])
    open_notion = col1.button("🚀 Open in notion")  # logic handled further down
    open_github = col2.button("💻 Github")  # logic handled further down

    
    if open_notion:
        open_link('https://bottlenose-client-925.notion.site/GAN-ji-0f10fdc083a244f5b15a97dd1c797d71')
    if open_github:
        open_link('https://github.com/GAN-ji')


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
    ###########################
    del checkpoint
    #########################
    torch.cuda.empty_cache()
    
    
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
        
    if 'samples_z' in st.session_state:
        st.session_state['latent'] = torch.unsqueeze(st.session_state['samples_z'][row_number-1][col_number-1], dim=0)
        st.session_state['latent'] = g_ema.get_latent(st.session_state['latent'])
        img, _ = g_ema(
        [st.session_state.latent],
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

    col1, col2 = st.columns([2,2])
    with col1:  
        Feature1 = st.slider('Expression 1', -10, 10, step=1, value=0)
        Feature2 = st.slider('Expression 2', -10, 10, step=1, value=0)
        Feature3 = st.slider('Color 1', -10, 10, step=1, value=0)
        Feature4 = st.slider('Color 2', -10, 10, step=1, value=0)
        Feature5 = st.slider('???', -10, 10, step=1, value=0)
    features = [Feature1, Feature2, Feature3, Feature4, Feature5]

    if sum(features)!=0 and 'samples_z' in st.session_state:
        idx_list = [1,2,8,9, 5]
        for i in range(len(idx_list)): 
            st.session_state['latent'] = sefa(idx_list[i], features[i], st.session_state['latent'], eigvec, trunc, g_ema)

    with col2:
        if 'latent' in st.session_state:
            image = Image.open('output.png')
            st.image(image, caption='Input Image', use_column_width=True)


    col1,col2,col3 = st.columns([3,2,2])
    with col2:
        if 'latent' in st.session_state:
            with open("output.png", "rb") as file:
                btn = st.download_button(
                    label="Download GAN-ji",
                    data=file,
                    file_name="GAN-ji_sample.png",
                    mime="image/png"
                )

                
    st.markdown('---')
    st.header('Emojify a person!')
    import numpy as np

    from model_ import Generator_
    #from util import *
    truncation = .5
#     generator1 = Generator_(256, 512, 8, channel_multiplier=2).eval().to(device)
#     generator2 = Generator_(256, 512, 8, channel_multiplier=2).to(device).eval()
    if 'generator1' not in st.session_state:
        st.session_state.generator1 = Generator_(256, 512, 8, channel_multiplier=2).eval().to(device)
    if 'generator2' not in st.session_state:
        st.session_state.generator2 = Generator_(256, 512, 8, channel_multiplier=2).to(device).eval()

    if 'mean_latent1' not in st.session_state:
        st.session_state.mean_latent1 = load_model(st.session_state.generator1, 'networks/ffhq256.pt')
    if 'mean_latent2' not in st.session_state:
        st.session_state.mean_latent2 = load_model(st.session_state.generator2, 'networks/635000.pt')
    torch.cuda.empty_cache()

    col1, col2 = st.columns([2,2])
    with col1:
        button1 = st.button("Random Face")
    with col2:
        button2 = st.button("Random Emoji")
        
    if button1:
        source_code = torch.randn([1, 512]).to(device)
        st.session_state.latent1 = st.session_state.generator1.get_latent(source_code, truncation=truncation, mean_latent=st.session_state.mean_latent1)
        source_im, _ = st.session_state.generator1(st.session_state.latent1)
        utils.save_image(
            source_im,
            f"latent1.png",
            normalize=True,
            range=(-1, 1),
            nrow=1,
        )
    if button2:
        reference_code = torch.randn([1, 512]).to(device)
        st.session_state.latent2 = st.session_state.generator2.get_latent(reference_code, truncation=truncation, mean_latent=st.session_state.mean_latent2)
        reference_im, _ = st.session_state.generator2(st.session_state.latent2)
        utils.save_image(
            reference_im,
            f"latent2.png",
            normalize=True,
            range=(-1, 1),
            nrow=1,
        )
    with col1:
        if 'latent1' in st.session_state:
            img1 = Image.open('latent1.png')
            st.image(img1, caption='Face Image', use_column_width=True)
    with col2:
        if 'latent2' in st.session_state:
            img2 = Image.open('latent2.png')
            st.image(img2, caption='Emoji Image', use_column_width=True)
    
    
    
    def toonify(latent1, latent2):
            with torch.no_grad():
                noise1 = [getattr(st.session_state.generator1.noises, f'noise_{i}') for i in range(st.session_state.generator1.num_layers)]
                noise2 = [getattr(st.session_state.generator2.noises, f'noise_{i}') for i in range(st.session_state.generator2.num_layers)]

                out1 = st.session_state.generator1.input(latent1[0])
                out2 = st.session_state.generator2.input(latent2[0])
                out = (1-early_alpha)*out1 + early_alpha*out2

                out1, _ = st.session_state.generator1.conv1(out, latent1[0], noise=noise1[0])
                out2, _ = st.session_state.generator2.conv1(out, latent2[0], noise=noise2[0])
                out = (1-early_alpha)*out1 + early_alpha*out2

                skip1 = st.session_state.generator1.to_rgb1(out, latent1[1])
                skip2 = st.session_state.generator2.to_rgb1(out, latent2[1])
                skip = (1-early_alpha)*skip1 + early_alpha*skip2

                i = 2
                for conv1_1, conv1_2, noise1_1, noise1_2, to_rgb1, conv2_1, conv2_2, noise2_1, noise2_2, to_rgb2 in zip(
                    st.session_state.generator1.convs[::2], st.session_state.generator1.convs[1::2], noise1[1::2], noise1[2::2], st.session_state.generator1.to_rgbs,
                    st.session_state.generator2.convs[::2], st.session_state.generator2.convs[1::2], noise2[1::2], noise2[2::2], st.session_state.generator2.to_rgbs
                ):


                    conv_alpha = early_alpha if i < num_swap else alpha
                    out1, _ = conv1_1(out, latent1[i], noise=noise1_1)
                    out2, _ = conv2_1(out, latent2[i], noise=noise2_1)
                    out = (1-conv_alpha)*out1 + conv_alpha*out2
                    i += 1

                    conv_alpha = early_alpha if i < num_swap else alpha
                    out1, _ = conv1_2(out, latent1[i], noise=noise1_2)
                    out2, _ = conv2_2(out, latent2[i], noise=noise2_2)
                    out = (1-conv_alpha)*out1 + conv_alpha*out2
                    i += 1

                    conv_alpha = early_alpha if i < num_swap else alpha
                    skip1 = to_rgb1(out, latent1[i], skip)
                    skip2 = to_rgb2(out, latent2[i], skip)
                    skip = (1-conv_alpha)*skip1 + conv_alpha*skip2

                    i += 1

            image = skip.clamp(-1,1)

            return image
    
    
    col1, col2, col3 = st.columns([3,2,2])
    if 'latent1' in st.session_state and 'latent2' in st.session_state:
        with col2:
            button = st.button("Emojify!")
        if button:
            g_ema=g_ema.cpu()
            num_swap =  3
            alpha =  1

            early_alpha = 0

            result = toonify(st.session_state.latent1, st.session_state.latent2)
            utils.save_image(
                result,
                f"emojify.png",
                normalize=True,
                range=(-1, 1),
                nrow=1,
            )
            if os.path.isfile('emojify.png'):
                emojify = Image.open('emojify.png')
                st.image(emojify, caption='Emojified Image', use_column_width=True)

