import tensorflow as tf
import os, sys
sys.path.append(os.path.abspath("."))

from data.load_pokemon_imgs import get_pokemon_images
import networks.WGAN_HUE as WGAN

img_dir = 'D:\\pokesprite\\pokemon-gen7x\\regular\\'
img_ds = get_pokemon_images(img_dir, (56, 68, 4), True)

wgan = WGAN.PokeWGAN((100,), (128, 56, 68, 4), './OUTPUT/pokeGAN_5/')

wgan.train(img_ds, 1000, 0, 1, 5)
