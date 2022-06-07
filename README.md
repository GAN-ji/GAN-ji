# GAN-ji

## `Introduction`
<p align='center'><b> ✨ Inference using Streamlit ✨</b></p> 
<p align='center'><img src='/asset/GAN-ji-gif(1).gif?raw=1' width = '900' ></p>

- 2022.04.25 ~ 2022.06.10
- 소속: Aiffel 양재캠퍼스
- 유례가 없는 팬데믹으로 인한 디지털 가속화, 국내 7000억 규모의 이모티콘 시장에서 이모지가 디지털 미디어를 통한 소통에 아주 중요한 역할을 한다. 하지만 제한된 종류로 의사소통의 한계가 명확하다는 점, 개인을 위한 맞춤 이모티콘 서비스의 부재, 그리고 궁금증과 흥미 등의 이유로 GAN을 이용한 맞춤 이모티콘 생성 서비스 배포를 팀 프로젝트로 진행하였다.
scratch부터 훈련, 사전학습된 얼굴 생성 모델 그리고 여러 시도들을 통해 훈련을 진행했다. joypixel의 emoji dataset을 super resolution을 통해 128부터 256, 512, 1024까지 여러 사이즈로 훈련을 시도했고, dataset도 여러 번 수정하여 1500~3000장 사이로 사용했다.훈련 환경은 GCP를 통한  V100(1개)과 구글 코랩 P100(1개) 두 가지를 이용하였다. 최종적으로 서비스 구현에 사용한 것은 256x256 해상도에서 전이학습을 한 것을 사용했다.

## `Data Preparation`

- 가장 기본적인 형태의 이모지를 사용하기 위해 joypixel의 다양한 dataset(링크걸기)을 훈련 목적에 맞게 수정하여 사용했다. 필요한 해상도로 dataset을 만들기 위해 waifu2x 초해상도(링크걸기)를 이용하여 upscale했다.

## `StyleGAN`

## `In-Progress`
### 1. 
<p align='center'><img src='/asset/GAN-ji-gif(2).gif?raw=1' width = '900' ></p>

### 2. 
<p align='center'>
  <img src='/asset/emojify.gif?  width='100' height='100'>
  <img src='/asset/emojify(2).gif? width='100' height='100'>
  <img src='/asset/emojify(3).gif? width='100' height='100'>
                                                           </p align='center'>
                                                           
## `completed`

                                                                             
## `Requirements`
- PyTorch 1.9.1
- CUDA 11.1
- streamlit 1.4.0
                                                                             
                                                                             
## `References`
- https://github.com/rosinality/stylegan2-pytorch
- 
