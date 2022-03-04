# POKEMON GAN

This is some work on some pokemon-based GAN models:

- There is the WGAN_Base model, which seeks to generate images of pokemon from a dataset  

- There is the WGAN_Colorize model, which seeks to colorize images of pokemon given a hue.

### __WGAN\_BASE__

The WGAN_Base model is a simple network using a network with a ResNet-based generator and a Pix2Pix-based discriminator, implemeted akin to the first iteration of StarGAN. It is trained using a Simple Wasserstein Loss with a Gradient Penalty on the Critic.

### __WGAN\_COLORIZE__

The WGAN_Colorize network is an iteration on the base model, but instead of generating new pokemon, it seeks to colorize existing pokemon given the pokemon's average hue. It does this by adding a couple of new losses to the generator:
-  a Hue Loss, which takes the Mean Square Error between the average hue of the image and the desired hue given to the generator, accounting for the fact that hues are cyclical and 1.0 is essentially the same as 0.0
- a Reconstructive Loss, which takes the Mean Absolute Error between the image and the reconstruction of the image, after the Black and White image is given to the Generator with the base image's average hue

### __WGAN\_Recolor__

This network is an iteration upon colorize, where instead of the network attempting to colorize a black and white image, the network instead recolors a full-color image, using the existing colors as a guide


## Datasets Used

These models were used with two seperate datasets, a dataset of low-quality Sprite-Based Pokemon Images, and a Dataset of High-Quality Pokemon Art.

Pokemon Sprites: https://github.com/msikma/pokesprite 

High Quality Pokemon: https://www.kaggle.com/kvpratama/pokemon-images-dataset