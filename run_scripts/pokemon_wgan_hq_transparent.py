import tensorflow as tf
import os, sys
sys.path.append(os.path.abspath("."))

from data.load_hq_pokemon_imgs import get_pokemon_images
import networks.WGAN_Base as WGAN

img_dir = 'D:\\pokemon_hq\\pokemon\\pokemon\\'
img_ds = get_pokemon_images(img_dir, (256, 256, 4), 1, False, False, True).unbatch()

wgan = WGAN.PokeWGAN([10000], (256, 256, 4), './OUTPUT/pokeGAN_color_hq_tp/', log_tiling=(2, 4))
cache_dir = os.path.join('./cached_datasets/pokeGAN_color_hq/', '{}/'.format(wgan.birthday))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
combined_ds = img_ds.cache(cache_dir).shuffle(5000, reshuffle_each_iteration=True).batch(8)

wgan.train(combined_ds, 1000, 0, 1, 10)
