# GAN-ji

## `Introduction`
<p align='center'><b> ✨ Inference using Streamlit ✨</b></p> 
<p align='center'><img src='/asset/GAN-ji-gif(1).gif?raw=1' width = '900' ></p>

- 2022.04.25 ~ 2022.06.10
- 소속: Aiffel 양재캠퍼스
- 유례가 없는 팬데믹으로 인한 디지털 가속화, 국내 7000억 규모의 이모티콘 시장에서 이모지가 디지털 미디어를 통한 소통에 아주 중요한 역할을 한다. 하지만 제한된 종류로 의사소통의 한계가 명확하다는 점, 개인을 위한 맞춤 이모티콘 서비스의 부재, 그리고 궁금증과 흥미 등의 이유로 GAN을 이용한 맞춤 이모티콘 생성 서비스 배포를 팀 프로젝트로 진행하였다.
scratch부터 훈련, 사전학습된 얼굴 생성 모델 그리고 여러 시도들을 통해 훈련을 진행했다. joypixel의 emoji dataset을 super resolution을 통해 128부터 256, 512, 1024까지 여러 사이즈로 훈련을 시도했고, dataset도 여러 번 수정하여 1500~3000장 사이로 사용했다.훈련 환경은 GCP를 통한  V100(1개)과 구글 코랩 P100(1개) 두 가지를 이용하였다. 최종적으로 서비스 구현에 사용한 것은 256x256 해상도에서 전이학습을 한 것을 사용했다.

## `Data Preparation`

- 가장 기본적인 형태의 이모지를 사용하기 위해 [joypixel의 다양한 dataset](https://www.joypixels.com/download,"dataset")을 훈련 목적에 맞게 수정하여 사용했다. 필요한 해상도로 dataset을 만들기 위해 [waifu2x 초해상도](https://github.com/tsurumeso/waifu2x-chainer)를 이용하여 upscale했다.

## NVlabs StyleGAN2 ADA
#### (+ FreezeD and hyper parameters)
프로젝트는 모두 StyleGAN2 ADA 위에서 이루어졌고, 처음 훈련을 시작할 때, NVlabs의 코드를 사용했다. Dataset이 굉장히 한정적이고, 굉장히 단순하기 때문에 overfitting이 매우 쉽게 일어날 것으로 보고, StyleGAN2 ADA를 처음부터 사용하였다. Tensorflow version과 pytorch version 모두 사용해보았는데 이에 대한 차이는 확인되지 않았다. Overfitting을 최대한 막기 위해 data augmentation인 하이퍼 파라미터 중 하나인 mirrored를 이용했고, 전이학습에서 Discriminator의 layer를 얼려서 훈련시키는 freeze D도 사용해보았다.

![image](https://user-images.githubusercontent.com/95264469/172424276-1dd23722-afd3-4074-82dc-bc622c3368c3.png)

아주 초반부에는 모자, 얼굴의 각도 또는 배경 등 제법 재미있는 결과도 보였지만 급속도로 모드 붕괴 현상을 보였다. 더 높은 해상도에서 좋은 결과가 나오는 편이라고 하나 결과는 크게 다르지 않았다. 아래 결과를 통해 overfitting이나 leaking 현상이 심하다는 것도 확인 할 수 있었다. FID 점수는 200점 밑으로 떨어진 적 없이 계속 높은 값을 기록했다.

<table>
  <tr>
    <td><img alt="" src='/asset/NVlabs StyleGAN2 ADA(1).png?' /></td><td><img alt="" src='/asset/NVlabs StyleGAN2 ADA(2).png?' /></td>
  <tr>
</table>

## Rosinality StyleGAN2 ADA
![image](https://user-images.githubusercontent.com/95264469/172443161-c9b3d78c-6a0f-473e-a8f0-f3e4f53a64b7.png)

Custom dataset을 이용한 타 프로젝트에서 많이 사용된 rosinality의 StyleGAN2 ADA를 사용하였다. 흥미롭게도 훈련이 안정적으로 진행되었고 훈련의 결과가 질적으로 매우 향상되었다. 전이학습이 아닌 축소한 모델과 축소한 latent vector로 밑바닥에서 부터(220k) 훈련한 결과도 준수하게 나왔는데, 다른 dataset에서 훈련을 진행하여 정확한 비교를 하기는 무리가 있다.(FID : 39.00) 하이퍼 파라미터들은 d_reg, freeze D, latent vector 등을 조절하며 훈련을 시도해보았는데, mixing을 미적용하여 훈련했을 때 결과의 해상도가 미세하게 상승하여 팀 내 정성 평가 및 FID 점수 모두 가장 좋은 결과를 기록했다.

![FID](https://user-images.githubusercontent.com/95264469/172443372-16cf0667-f671-4aab-8fb0-3a246e75f3f9.png)


## Closed Form Factorization

![testing_customizer](https://user-images.githubusercontent.com/95264469/172446971-d63eb273-e08f-42af-8634-dceb32a7c365.gif)

Latent Space에서의 변화는 이미지에 다양한 영향을 미친다. 아직 자세하게 탐구가 된 부분은 아니지만, Latent Space에서의 이미지 조작을 하기 위해 다양한 시도를 하고 있다. 
Closed Form Factorization은 비지도 학습을 통해 Latent Space에서 의미 있는 변화 방향을 찾는다. 이 방법을 이용하면 다음과 같이 찾은 방향을 따라 이동하면 GAN-ji의 특징이 변한다. 

![factor_index-5_degree-3 5](https://user-images.githubusercontent.com/95264469/172442977-7aca5606-7483-42e2-94ad-c60587cad164.png)


## Emojify(Network Blending)


<p align='center'> 
  <img src='/asset/emojify.gif?  width='100' height='100'>
  <img src='/asset/emojify(2).gif? width='100' height='100'>
  <img src='/asset/emojify(3).gif? width='100' height='100'>
</p align='center'>
<p align='center'><b> generated ffhq to emoji </b></p>
                  
                 
<p align='center'> 
  <img src='/asset/emojify(4).gif?  width='100' height='100'>
</p align='center'>
<p align='center'><b> projected 공유 to emoji </b></p>


                                                                             
전이학습을 통해 특정 데이터에 학습된 모델을 다른 도메인의 데이터에 이어서 학습을 하면, 두 모델 사이에 어느 정도의 공통점이 존재하게 된다. 두 모델의 각 레이어에서 적절하게 가중치를 swapping을 해주면, 자연스럽게 두 도메인을 섞을 수 있다. 다음과 같이 이모지의 형태를 유지하며 사람의 특징을 가져오는 식으로 face 도메인과 emoji 도메인의 사이의 이미지를 생성할 수 있다. 

## Conclusion
- 사이즈 256 FFHQ(550k) 모델에서 전이학습을 하여 100k의 훈련을 통해 만족스러운 결과를 낼 수 있엇다. 결과적으로 같은 SG2 ADA 모델에서 전혀 다른 결과가 나왔는데, GAN 훈련의 기묘함을 느끼는 동시에 NVlabs는 튜닝 가능한 여러 값들이 FFHQ나 AHQ 같은 데이터에 더욱 최적화되어 있어서 이러한 결과가 나온 것으로 보인다. 
- Streamlit을 통해 랜덤 이모티콘 생성, Feature customize(Closed form factorization), Emojify(Network blending) 3가지 기능을 구현하여 서비스 배포를 하였다.
- 개인의 사진을 넣어 projector를 통과시켜 emojify를 시키는 것이 가장 구현하고자 하는 기능이었지만 projector를 통과시킬 때 5분 이상 소요되는 점과 결과물로 출력되는 이모티콘이 서비스로서의 가치가 떨어진다는 점에서 배포 단계에서는 제외시켰다. 기획한 서비스의 상품성을 위해서는 projector 문제를 극복해야함은 물론 도메인 차이로 발생하는 저품질의 결과물을 개선해야 하는 숙제들을 해결해야 한다.

                                                                             
## `Requirements`
- PyTorch 1.9.1
- CUDA 11.1
- streamlit 1.4.0
                                                                             
                                                                             
## `References`
- [침착한 생성모델](https://github.com/bryandlee/malnyun_faces)
- [Cartoon-StyleGAN](https://github.com/happy-jihye/Cartoon-StyleGAN)
