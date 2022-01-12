import tensorflow as tf
import os, sys
sys.path.append(os.path.abspath("."))

from data.load_hq_pokemon_imgs import get_pokemon_images
import networks.WGAN_Base as WGAN

img_dir = 'D:\\pokemon_hq\\pokemon_jpg\\pokemon_jpg\\'
img_ds = get_pokemon_images(img_dir, (256, 256, 3), 1, False, False, True).unbatch()

wgan = WGAN.PokeWGAN((100, 100), (256, 256, 3), './OUTPUT/pokeGAN_color_hq/', log_tiling=(2, 4))
cache_dir = os.path.join('./cached_datasets/pokeGAN_color_hq/', '{}/'.format(wgan.birthday))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
combined_ds = img_ds.cache(cache_dir).shuffle(5000).batch(8)

wgan.train(combined_ds, 1000, 0, 1, 10)
