# 03_PlayWithGANs

## Incremental Pix2Pix

### Method

In this part, a conditional GAN is trained to map segment image to rgb image. The generator is the same as the generator in Assignment 2, and the discriminator is also implemented using CNN. In the last layer, a fully connected network is used to map multi channels to a one dimention probability.

To train a conditional GAN, it's necessary to change the loss function for L1 to cGAN loss. The origin loss is $$\mathcal{L}_{cGAN}=\mathbf{E}_{x,y}[logD(x, y)]+\mathbf(E)_{x,z}[log(1-D(x,G(x,z))]$$

And it is the same as the binary cross entropy loss in this case, so I use this as the loss function.

### run

To train the condition GAN, just run

```
cd Assignments/03_PlayWithGANs/Pix2Pix
python train.py
```
You can also 