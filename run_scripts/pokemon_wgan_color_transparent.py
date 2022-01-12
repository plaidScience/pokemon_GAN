import tensorflow as tf
import os, sys
sys.path.append(os.path.abspath("."))

from data.load_pokemon_imgs import get_pokemon_images
import networks.WGAN_Base as WGAN

img_dir = 'D:\\pokesprite\\pokemon-gen7x\\regular\\'
img_ds = get_pokemon_images(img_dir, (56, 68, 4), True, (48, 48), False, False, True).unbatch()
shiny_dir = "D:\\pokesprite\\pokemon-gen7x\\shiny\\"
shiny_ds = get_pokemon_images(shiny_dir, (56, 68, 4), True, (48, 48), False, False, True).unbatch()

print(img_ds, shiny_ds)

wgan = WGAN.PokeWGAN((10000,), (48, 48, 4), './OUTPUT/pokeGAN_color_tp/')
cache_dir = os.path.join('./cached_datasets/pokeGAN_color_tp/', '{}/'.format(wgan.birthday))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
combined_ds = img_ds.concatenate(shiny_ds).cache(cache_dir).shuffle(5000).batch(128)
print(combined_ds)
wgan.train(combined_ds, 10000, 0, 5, 50)
