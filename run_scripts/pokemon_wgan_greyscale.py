import tensorflow as tf
import os, sys
sys.path.append(os.path.abspath("."))

from data.load_pokemon_imgs import get_pokemon_images
import networks.WGAN_Base as WGAN

img_dir = 'D:\\pokesprite\\pokemon-gen7x\\regular\\'
img_ds = get_pokemon_images(img_dir, (56, 68, 4), True, (40, 40), True, True, True).unbatch()
shiny_dir = "D:\\pokesprite\\pokemon-gen7x\\shiny\\"
shiny_ds = get_pokemon_images(shiny_dir, (56, 68, 4), True, (40, 40), True, True, True).unbatch()

wgan = WGAN.PokeWGAN((100,), (128, 40, 40, 1), './OUTPUT/pokeGAN_grey/')
cache_dir = os.path.join('./cached_datasets/pokeGAN_grey/', '{}/'.format(wgan.birthday))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
combined_ds = img_ds.concatenate(shiny_ds).cache(cache_dir).shuffle(5000, reshuffle_each_iteration=True).batch(512)

wgan.train(combined_ds, 4000, 0, 5, 50)
