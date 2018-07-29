# S2Dnet

[Specular-to-Diffuse Translation for Multi-View Reconstruction](https://www.dropbox.com/s/ouvm6hh7ge5zdwy/S2Dnet.pdf?dl=0) 
ECCV, 2018 

<p align="center"><img width="80%" src="git_img/teaser.png" /></p>

<p align="center"><img width="100%" src="git_img/network.png" /></p>

<br/>


<br/>

## Dependencies

* [Python](https://www.continuum.io/downloads)
* [PyTorch](http://pytorch.org/)
* [TensorFlow](https://www.tensorflow.org/) (optional for tensorboard)
* [PBRT](http://pbrt.org/scenes-v3.html) (optional for rendering)
* [CycleGAN](https://github.com/junyanz/CycleGAN) (our code is based on this implementation)

### Downloading (Dropbox links)

* [Full multi-view synthetic training data (172 GB)](https://www.dropbox.com/s/0l146k934t8tqqi/huge_uni_render_rnn.zip?dl=0)
* [Tiny multi-view synthetic training data (3 GB)](https://www.dropbox.com/s/uv8onade36v6pto/tiny_uni_render_rnn.zip?dl=0)
* [Real training data](https://www.dropbox.com/s/mnvhit9a9ftuxp0/output_color_multi_2.zip?dl=0) 
* [Some Synthetic testing data](https://www.dropbox.com/s/zfd8p5qwwolr6yx/test_data_rendered.zip?dl=0) 
* [Some Real testing data](https://www.dropbox.com/s/xcc0ywhcb5nuntz/test_data_real.zip?dl=0) 
* [Aligned 3D models for rendering](https://www.dropbox.com/s/jno3g7867ysvsy6/geometry.zip?dl=0) 
* [Environment maps for rendering](https://www.dropbox.com/s/13asq10w7x6vame/textures.zip?dl=0) 


### Training example

```bash
$ python train.py --dataroot ../huge_uni_render_rnn --logroot ./logs/job101CP --name job_submit_101C_re1_pixel --model cycle_gan --no_dropout --loadSize 512 --fineSize 512 --patchSize 256 --which_model_netG unet_512_Re1 --which_model_netD patch_512_256_multi_new --lambda_A 10 --lambda_B 10 --lambda_vgg 5 --norm pixel
```


### Testing

Please refer to "./useful_scripts/evaluation/"

Scripts of SIFT, SMVS, and rendering are in "./useful_scripts/evaluation/". 

Please contact the author for detailed instruction.


