import tensorflow as tf
import os, sys
sys.path.append(os.path.abspath("."))

from data.load_pokemon_imgs import get_pokemon_images
import networks.WGAN_Colorize as WGAN

LOAD_WIDTH = 112//2
LOAD_HEIGHT = 136//2

CROP_DIMS = 96//2

img_dir = 'D:\\pokesprite\\pokemon-gen7x\\regular\\'
img_ds = get_pokemon_images(img_dir, (LOAD_WIDTH, LOAD_HEIGHT, 4), True, (CROP_DIMS, CROP_DIMS), False, False, True).unbatch()
shiny_dir = "D:\\pokesprite\\pokemon-gen7x\\shiny\\"
shiny_ds = get_pokemon_images(shiny_dir, (LOAD_WIDTH, LOAD_HEIGHT, 4), True, (CROP_DIMS, CROP_DIMS), False, False, True).unbatch()

wgan = WGAN.PokeWGAN((CROP_DIMS, CROP_DIMS, 4), './OUTPUT/pokeGAN_colorize_tp/', log_tiling=(4, 8), plot_scaling=3)
cache_dir = os.path.join('./cached_datasets/pokeGAN_colorize_tp/', '{}/'.format(wgan.birthday))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
combined_ds = img_ds.concatenate(shiny_ds).cache(cache_dir).shuffle(5000, reshuffle_each_iteration=True).batch(32)


wgan.train(combined_ds, 5000, 0, 1, 10)
