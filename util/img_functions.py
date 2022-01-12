import tensorflow as tf
def get_avg_hue(img):
    avg_rgb = tf.reduce_mean(img, axis=[0, 1])
    avg_hsv = tf.image.rgb_to_hsv(avg_rgb)
    avg_hue = avg_hsv[0]
    return avg_hue