import tensorflow as tf
import os, sys
sys.path.append(os.path.abspath("."))

from data.load_hq_pokemon_imgs import get_pokemon_images
import networks.WGAN_Recolor as WGAN

IMAGE_SHAPE = (256, 256, 3)

img_dir = 'D:\\pokemon_hq\\pokemon_jpg\\pokemon_jpg\\'
img_ds = get_pokemon_images(img_dir, IMAGE_SHAPE, 1, False, False, True).unbatch()

wgan = WGAN.PokeWGAN(IMAGE_SHAPE, './OUTPUT/pokeGAN_recolor_hq/', log_tiling=(2, 2))
cache_dir = os.path.join('./cached_datasets/pokeGAN_recolor_hq/', '{}/'.format(wgan.birthday))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
combined_ds = img_ds.cache(cache_dir).shuffle(5000, reshuffle_each_iteration=True).batch(4)

wgan.train(combined_ds, 5000, 0, 1, 10)
