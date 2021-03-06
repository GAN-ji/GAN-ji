<div align="right">
  
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FGAN-ji&count_bg=%23A6D2FE&title_bg=%23555555&icon=github.svg&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
<a href="https://modulabs.co.kr/">![](https://img.shields.io/badge/-MODULABS-white)</a>
<a href="https://github.com/mybloodykeyboard">![](https://img.shields.io/static/v1?label=๐&message=๋ฐ์ ์ฐ&color=blueviolet)</a>
<a href="https://github.com/kim-seo-hyun">![](https://img.shields.io/static/v1?label=๐&message=๊น์ํ&color=ff69b4)</a>
<a href="https://github.com/paulcho98">![](https://img.shields.io/static/v1?label=๐&message=์กฐํ๋น&color=brightgreen)</a>
</div>


## `Introduction`
<p align='center'><b> โจ Inference using Streamlit โจ</b></p> 
<p align='center'><img src='/asset/streamlit_final.gif?raw=1' width = '900' ></p>

- 2022.04.25 ~ 2022.06.10
- Aiffel YJ AIFFELTHON
- Team GAN-ji
- ํ์ฅ: ๋ฐ์ ์ฐ, ํ์: ๊น์ํ, ์กฐํ๋น
- ์ ๋ก๊ฐ ์๋ ํฌ๋ฐ๋ฏน์ผ๋ก ์ธํ ๋์งํธ ๊ฐ์ํ, ๊ตญ๋ด 7000์ต ๊ท๋ชจ์ ์ด๋ชจํฐ์ฝ ์์ฅ์์ ์ด๋ชจ์ง๊ฐ ๋์งํธ ๋ฏธ๋์ด๋ฅผ ํตํ ์ํต์ ์์ฃผ ์ค์ํ ์ญํ ์ ํ๋ค. ํ์ง๋ง ์ ํ๋ ์ข๋ฅ๋ก ์์ฌ์ํต์ ํ๊ณ๊ฐ ๋ชํํ๋ค๋ ์ , ๊ฐ์ธ์ ์ํ ๋ง์ถค ์ด๋ชจํฐ์ฝ ์๋น์ค์ ๋ถ์ฌ, ๊ทธ๋ฆฌ๊ณ  ๊ถ๊ธ์ฆ๊ณผ ํฅ๋ฏธ ๋ฑ์ ์ด์ ๋ก GAN์ ์ด์ฉํ ๋ง์ถค ์ด๋ชจํฐ์ฝ ์์ฑ ์๋น์ค ๋ฐฐํฌ๋ฅผ ํ ํ๋ก์ ํธ๋ก ์งํํ์๋ค.
scratch๋ถํฐ ํ๋ จ, ์ฌ์ ํ์ต๋ ์ผ๊ตด ์์ฑ ๋ชจ๋ธ ๊ทธ๋ฆฌ๊ณ  ์ฌ๋ฌ ์๋๋ค์ ํตํด ํ๋ จ์ ์งํํ๋ค. joypixel์ emoji dataset์ super resolution์ ํตํด 128๋ถํฐ 256, 512, 1024๊น์ง ์ฌ๋ฌ ์ฌ์ด์ฆ๋ก ํ๋ จ์ ์๋ํ๊ณ , dataset๋ ์ฌ๋ฌ ๋ฒ ์์ ํ์ฌ 1500~3000์ฅ ์ฌ์ด๋ก ์ฌ์ฉํ๋ค.ํ๋ จ ํ๊ฒฝ์ GCP๋ฅผ ํตํ  V100(1๊ฐ)๊ณผ ๊ตฌ๊ธ ์ฝ๋ฉ P100(1๊ฐ) ๋ ๊ฐ์ง๋ฅผ ์ด์ฉํ์๋ค. ์ต์ข์ ์ผ๋ก ์๋น์ค ๊ตฌํ์ ์ฌ์ฉํ ๊ฒ์ 256x256 ํด์๋์์ ์ ์ดํ์ต์ ํ ๊ฒ์ ์ฌ์ฉํ๋ค.

## `Data Preparation`

- ๊ฐ์ฅ ๊ธฐ๋ณธ์ ์ธ ํํ์ ์ด๋ชจ์ง๋ฅผ ์ฌ์ฉํ๊ธฐ ์ํด [joypixel์ ๋ค์ํ dataset](https://www.joypixels.com/download,"dataset")์ ํ๋ จ ๋ชฉ์ ์ ๋ง๊ฒ ์์ ํ์ฌ ์ฌ์ฉํ๋ค. ํ์ํ ํด์๋๋ก dataset์ ๋ง๋ค๊ธฐ ์ํด [waifu2x ์ดํด์๋](https://github.com/tsurumeso/waifu2x-chainer)๋ฅผ ์ด์ฉํ์ฌ upscaleํ๋ค.

## `NVlabs StyleGAN2 ADA`
#### (+ FreezeD and hyper parameters)
ํ๋ก์ ํธ๋ ๋ชจ๋ StyleGAN2 ADA ์์์ ์ด๋ฃจ์ด์ก๊ณ , ์ฒ์ ํ๋ จ์ ์์ํ  ๋, NVlabs์ ์ฝ๋๋ฅผ ์ฌ์ฉํ๋ค. Dataset์ด ๊ต์ฅํ ํ์ ์ ์ด๊ณ , ๊ต์ฅํ ๋จ์ํ๊ธฐ ๋๋ฌธ์ overfitting์ด ๋งค์ฐ ์ฝ๊ฒ ์ผ์ด๋  ๊ฒ์ผ๋ก ๋ณด๊ณ , StyleGAN2 ADA๋ฅผ ์ฒ์๋ถํฐ ์ฌ์ฉํ์๋ค. Tensorflow version๊ณผ pytorch version ๋ชจ๋ ์ฌ์ฉํด๋ณด์๋๋ฐ ์ด์ ๋ํ ์ฐจ์ด๋ ํ์ธ๋์ง ์์๋ค. Overfitting์ ์ต๋ํ ๋ง๊ธฐ ์ํด data augmentation์ธ ํ์ดํผ ํ๋ผ๋ฏธํฐ ์ค ํ๋์ธ mirrored๋ฅผ ์ด์ฉํ๊ณ , ์ ์ดํ์ต์์ Discriminator์ layer๋ฅผ ์ผ๋ ค์ ํ๋ จ์ํค๋ freeze D๋ ์ฌ์ฉํด๋ณด์๋ค.

<table>
    <td><img alt="" src='/asset/dataset.jpg?' /></td><td><img alt="" src='/asset/GAN-ji.jpg?' /></td>
</table>

์์ฃผ ์ด๋ฐ๋ถ์๋ ๋ชจ์, ์ผ๊ตด์ ๊ฐ๋ ๋๋ ๋ฐฐ๊ฒฝ ๋ฑ ์ ๋ฒ ์ฌ๋ฏธ์๋ ๊ฒฐ๊ณผ๋ ๋ณด์์ง๋ง ๊ธ์๋๋ก ๋ชจ๋ ๋ถ๊ดด ํ์์ ๋ณด์๋ค. ๋ ๋์ ํด์๋์์ ์ข์ ๊ฒฐ๊ณผ๊ฐ ๋์ค๋ ํธ์ด๋ผ๊ณ  ํ๋ ๊ฒฐ๊ณผ๋ ํฌ๊ฒ ๋ค๋ฅด์ง ์์๋ค. ์๋ ๊ฒฐ๊ณผ๋ฅผ ํตํด overfitting์ด๋ leaking ํ์์ด ์ฌํ๋ค๋ ๊ฒ๋ ํ์ธ ํ  ์ ์์๋ค. FID ์ ์๋ 200์  ๋ฐ์ผ๋ก ๋จ์ด์ง ์  ์์ด ๊ณ์ ๋์ ๊ฐ์ ๊ธฐ๋กํ๋ค.

<table>
    <td><img alt="" src='/asset/NVlabs StyleGAN2 ADA(1).png?' /></td><td><img alt="" src='/asset/NVlabs StyleGAN2 ADA(2).png?' /></td>
</table>

## `Rosinality StyleGAN2 ADA`
<p align='center'>
<img src='/asset/Rosinality StyleGAN2 ADA.png? raw=1' width = '500' ></p>
</p>

Custom dataset์ ์ด์ฉํ ํ ํ๋ก์ ํธ์์ ๋ง์ด ์ฌ์ฉ๋ rosinality์ StyleGAN2 ADA๋ฅผ ์ฌ์ฉํ์๋ค. ํฅ๋ฏธ๋กญ๊ฒ๋ ํ๋ จ์ด ์์ ์ ์ผ๋ก ์งํ๋์๊ณ  ํ๋ จ์ ๊ฒฐ๊ณผ๊ฐ ์ง์ ์ผ๋ก ๋งค์ฐ ํฅ์๋์๋ค. ์ ์ดํ์ต์ด ์๋ ์ถ์ํ ๋ชจ๋ธ๊ณผ ์ถ์ํ latent vector๋ก ๋ฐ๋ฐ๋ฅ์์ ๋ถํฐ(220k) ํ๋ จํ ๊ฒฐ๊ณผ๋ ์ค์ํ๊ฒ ๋์๋๋ฐ, ๋ค๋ฅธ dataset์์ ํ๋ จ์ ์งํํ์ฌ ์ ํํ ๋น๊ต๋ฅผ ํ๊ธฐ๋ ๋ฌด๋ฆฌ๊ฐ ์๋ค.(FID : 39.00) ํ์ดํผ ํ๋ผ๋ฏธํฐ๋ค์ d_reg, freeze D, latent vector ๋ฑ์ ์กฐ์ ํ๋ฉฐ ํ๋ จ์ ์๋ํด๋ณด์๋๋ฐ, mixing์ ๋ฏธ์ ์ฉํ์ฌ ํ๋ จํ์ ๋ ๊ฒฐ๊ณผ์ ํด์๋๊ฐ ๋ฏธ์ธํ๊ฒ ์์นํ์ฌ ํ ๋ด ์ ์ฑ ํ๊ฐ ๋ฐ FID ์ ์ ๋ชจ๋ ๊ฐ์ฅ ์ข์ ๊ฒฐ๊ณผ๋ฅผ ๊ธฐ๋กํ๋ค.

<p align='center'>
<img src='/asset/FID _score.png? raw=1' width = '500' ></p>
</p>

## `Closed Form Factorization`
<p align='center'>
<img src='/asset/customize your emoji.gif? raw=1' width = '600' ></p>
</p>

Latent Space์์์ ๋ณํ๋ ์ด๋ฏธ์ง์ ๋ค์ํ ์ํฅ์ ๋ฏธ์น๋ค. ์์ง ์์ธํ๊ฒ ํ๊ตฌ๊ฐ ๋ ๋ถ๋ถ์ ์๋์ง๋ง, Latent Space์์์ ์ด๋ฏธ์ง ์กฐ์์ ํ๊ธฐ ์ํด ๋ค์ํ ์๋๋ฅผ ํ๊ณ  ์๋ค. 
Closed Form Factorization์ ๋น์ง๋ ํ์ต์ ํตํด Latent Space์์ ์๋ฏธ ์๋ ๋ณํ ๋ฐฉํฅ์ ์ฐพ๋๋ค. ์ด ๋ฐฉ๋ฒ์ ์ด์ฉํ๋ฉด ๋ค์๊ณผ ๊ฐ์ด ์ฐพ์ ๋ฐฉํฅ์ ๋ฐ๋ผ ์ด๋ํ๋ฉด GAN-ji์ ํน์ง์ด ๋ณํ๋ค. 

<p align='center'>
<img src='/asset/Closed Form Factorization.png? raw=1' width = '600' ></p>
</p>


## `Emojify(Network Blending)`

<p align='center'> 
  <img src='/asset/emojify.gif?  width='100' height='100'>
  <img src='/asset/emojify(2).gif? width='100' height='100'>
  <img src='/asset/emojify(3).gif? width='100' height='100'>
</p align='center'>
<p align='center'><b> generated ffhq to emoji </b></p>
                  
                 
<p align='center'> 
  <img src='/asset/emojify(4).gif?  width='100' height='100'>
</p align='center'>
<p align='center'><b> projected ๊ณต์  to emoji </b></p>


                                                                             
์ ์ดํ์ต์ ํตํด ํน์  ๋ฐ์ดํฐ์ ํ์ต๋ ๋ชจ๋ธ์ ๋ค๋ฅธ ๋๋ฉ์ธ์ ๋ฐ์ดํฐ์ ์ด์ด์ ํ์ต์ ํ๋ฉด, ๋ ๋ชจ๋ธ ์ฌ์ด์ ์ด๋ ์ ๋์ ๊ณตํต์ ์ด ์กด์ฌํ๊ฒ ๋๋ค. ๋ ๋ชจ๋ธ์ ๊ฐ ๋ ์ด์ด์์ ์ ์ ํ๊ฒ ๊ฐ์ค์น๋ฅผ swapping์ ํด์ฃผ๋ฉด, ์์ฐ์ค๋ฝ๊ฒ ๋ ๋๋ฉ์ธ์ ์์ ์ ์๋ค. ๋ค์๊ณผ ๊ฐ์ด ์ด๋ชจ์ง์ ํํ๋ฅผ ์ ์งํ๋ฉฐ ์ฌ๋์ ํน์ง์ ๊ฐ์ ธ์ค๋ ์์ผ๋ก face ๋๋ฉ์ธ๊ณผ emoji ๋๋ฉ์ธ์ ์ฌ์ด์ ์ด๋ฏธ์ง๋ฅผ ์์ฑํ  ์ ์๋ค. 

## `Conclusion`
- ์ฌ์ด์ฆ 256 FFHQ(550k) ๋ชจ๋ธ์์ ์ ์ดํ์ต์ ํ์ฌ 100k์ ํ๋ จ์ ํตํด ๋ง์กฑ์ค๋ฌ์ด ๊ฒฐ๊ณผ๋ฅผ ๋ผ ์ ์์๋ค. ๊ฒฐ๊ณผ์ ์ผ๋ก ๊ฐ์ SG2 ADA ๋ชจ๋ธ์์ ์ ํ ๋ค๋ฅธ ๊ฒฐ๊ณผ๊ฐ ๋์๋๋ฐ, GAN ํ๋ จ์ ๊ธฐ๋ฌํจ์ ๋๋ผ๋ ๋์์ NVlabs๋ ํ๋ ๊ฐ๋ฅํ ์ฌ๋ฌ ๊ฐ๋ค์ด FFHQ๋ AHQ ๊ฐ์ ๋ฐ์ดํฐ์ ๋์ฑ ์ต์ ํ๋์ด ์์ด์ ์ด๋ฌํ ๊ฒฐ๊ณผ๊ฐ ๋์จ ๊ฒ์ผ๋ก ๋ณด์ธ๋ค. 
- Streamlit์ ํตํด ๋๋ค ์ด๋ชจํฐ์ฝ ์์ฑ, Feature customize(Closed form factorization), Emojify(Network blending) 3๊ฐ์ง ๊ธฐ๋ฅ์ ๊ตฌํํ์ฌ ์๋น์ค ๋ฐฐํฌ๋ฅผ ํ์๋ค.
- ๊ฐ์ธ์ ์ฌ์ง์ ๋ฃ์ด projector๋ฅผ ํต๊ณผ์์ผ emojify๋ฅผ ์ํค๋ ๊ฒ์ด ๊ฐ์ฅ ๊ตฌํํ๊ณ ์ ํ๋ ๊ธฐ๋ฅ์ด์์ง๋ง projector๋ฅผ ํต๊ณผ์ํฌ ๋ 5๋ถ ์ด์ ์์๋๋ ์ ๊ณผ ๊ฒฐ๊ณผ๋ฌผ๋ก ์ถ๋ ฅ๋๋ ์ด๋ชจํฐ์ฝ์ด ์๋น์ค๋ก์์ ๊ฐ์น๊ฐ ๋จ์ด์ง๋ค๋ ์ ์์ ๋ฐฐํฌ ๋จ๊ณ์์๋ ์ ์ธ์์ผฐ๋ค. ๊ธฐํํ ์๋น์ค์ ์ํ์ฑ์ ์ํด์๋ projector ๋ฌธ์ ๋ฅผ ๊ทน๋ณตํด์ผํจ์ ๋ฌผ๋ก  ๋๋ฉ์ธ ์ฐจ์ด๋ก ๋ฐ์ํ๋ ์ ํ์ง์ ๊ฒฐ๊ณผ๋ฌผ์ ๊ฐ์ ํด์ผ ํ๋ ์์ ๋ค์ ํด๊ฒฐํด์ผ ํ๋ค.
                                                                             
## `How to run this app`
It is suggested that creating a new virtual environment, then running:
```
  git clone https://github.com/GAN-ji/GAN-ji.git
  cd GAN-ji
  pip install -r requirements.txt
  streamlit run app.py
```
<b> We tested on Python 3.7.13, PyTorch 1.9.1, CUDA 11.1 </b>

                                                                             
                                                                             
## `References`
- [์นจ์ฐฉํ ์์ฑ๋ชจ๋ธ](https://github.com/bryandlee/malnyun_faces)
- [Cartoon-StyleGAN](https://github.com/happy-jihye/Cartoon-StyleGAN)
