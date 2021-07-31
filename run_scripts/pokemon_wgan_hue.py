import tensorflow as tf
import os, sys
sys.path.append(os.path.abspath("."))

from data.load_pokemon_imgs import get_pokemon_images
import networks.WGAN_HUE as WGAN

img_dir = 'D:\\pokesprite\\pokemon-gen7x\\regular\\'
img_ds = get_pokemon_images(img_dir, (56, 68, 4), True, (40, 40), True, True).unbatch()
shiny_dir = "D:\\pokesprite\\pokemon-gen7x\\shiny\\"
shiny_ds = get_pokemon_images(shiny_dir, (56, 68, 4), True, (40, 40), True, True).unbatch()

combined_ds = img_ds.concatenate(shiny_ds).shuffle(5000).batch(128)

wgan = WGAN.PokeWGAN((100,), (128, 40, 40, 3), './OUTPUT/pokeGAN/')

wgan.train(combined_ds, 1000, 0, 1, 5)
