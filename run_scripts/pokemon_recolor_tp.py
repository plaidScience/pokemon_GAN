import tensorflow as tf
import os, sys
sys.path.append(os.path.abspath("."))

from data.load_pokemon_imgs import get_pokemon_images
import networks.WGAN_Recolor as WGAN

LOAD_WIDTH = 112
LOAD_HEIGHT = 136

CROP_DIMS = 80

img_dir = 'D:\\pokesprite\\pokemon-gen7x\\regular\\'
img_ds = get_pokemon_images(img_dir, (LOAD_WIDTH, LOAD_HEIGHT, 4), True, (CROP_DIMS, CROP_DIMS), False, False, True).unbatch()
shiny_dir = "D:\\pokesprite\\pokemon-gen7x\\shiny\\"
shiny_ds = get_pokemon_images(shiny_dir, (LOAD_WIDTH, LOAD_HEIGHT, 4), True, (CROP_DIMS, CROP_DIMS), False, False, True).unbatch()

wgan = WGAN.PokeWGAN((CROP_DIMS, CROP_DIMS, 4), './OUTPUT/pokeGAN_recolor_tp/', log_tiling=(4, 8), plot_scaling=3)
combined_ds = img_ds.concatenate(shiny_ds).shuffle(5000, reshuffle_each_iteration=True).batch(32)


wgan.train(combined_ds, 5000, 0, 1, 10)
