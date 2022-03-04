from configparser import Interpolation
import tensorflow as tf
from math import floor

def get_pokemon_images(img_dir, img_shape=(56, 68, 3), resize_bottom=True, resize_shape=(40, 40), rgba2rgb=False, rgb2grey=False, norm=True):
    img_height = img_shape[-3]
    img_width = img_shape[-2]
    channels = img_shape[-1]
    if len(img_shape) > 3:
        batch_size = img_shape[-4]
    else:
        batch_size = 64
    if channels == 4:
        color_mode='rgba'
    elif channels == 1:
        color_mode='grayscale'
    elif channels==3:
        color_mode='rgb'
    poke_ds = tf.keras.preprocessing.image_dataset_from_directory(img_dir, label_mode=None, color_mode=color_mode, image_size=(img_height, img_width), batch_size=batch_size)
    if resize_bottom==True or rgba2rgb == True or rgb2grey==True or norm==True:
        if resize_bottom==True: print('Resizing With Cut From Bottom of Image')
        if rgba2rgb==True: print('Converting RGBA to RGB')
        if rgb2grey==True: print("Converting RGB to Grey")
        if norm==True: print('Normalizing Image')
        map_fn = lambda image: dataset_map(image, resize_bottom, (img_shape, resize_shape), rgba2rgb, rgb2grey, norm)
        poke_ds = poke_ds.map(map_fn)
    return poke_ds

def dataset_map(image, resize_bottom=True, resize_shape=((56, 68, 3), (40, 40)), rgba2rgb=False, rgb2grey=False, norm=True):
    if resize_bottom == True:
        img_shape, resize_shape = resize_shape
        image = tf.image.crop_to_bounding_box(image, img_shape[-3]-resize_shape[-2], floor((img_shape[-2]-resize_shape[-1])/2), resize_shape[-2], resize_shape[-1])
    if rgba2rgb == True:
        print(image.shape)
        image = rgba_to_rgb(image)
    if rgb2grey == True:
        print(image.shape)
        #assert image.shape[-1] == 3
        image = tf.image.rgb_to_grayscale(image)
    if norm ==True:
        image = image/127.5 - 1.0
    return image

def rgba_to_rgb(imgs):
    base_shape = tf.shape(imgs)
    shape = tf.concat([base_shape[:-1], [base_shape[-1]-1]], axis = 0)
    background = tf.ones(shape, dtype=tf.float32)*255
    mask = tf.split(imgs, 4, axis=-1)[-1]
    mask = tf.math.ceil(mask/255.0)
    start_slice = [0 for i in imgs.shape]
    rgb_image = tf.slice(imgs, start_slice, shape, name="rgb_image")
    output_img = rgb_image*mask + background*(1-mask)
    output_shape = [i for i in output_img.shape]
    return output_img


def main():
    img_dir = 'D:\\pokesprite\\pokemon-gen7x\\regular\\'
    img_ds = get_pokemon_images(img_dir)
    print(img_ds)
    print(img_ds.cardinality())
if __name__ == "__main__":
    main()
