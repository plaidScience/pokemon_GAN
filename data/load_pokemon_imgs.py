import tensorflow as tf

def get_pokemon_images(img_dir, img_shape=(56, 68, 3), norm=True):
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
    if norm==True:
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./127.5, offset=-1.0)
        poke_ds = poke_ds.map(normalization_layer)
    return poke_ds


def main():
    img_dir = 'D:\\pokesprite\\pokemon-gen7x\\regular\\'
    img_ds = get_pokemon_images(img_dir)
    print(img_ds)
    print(img_ds.cardinality())
if __name__ == "__main__":
    main()
