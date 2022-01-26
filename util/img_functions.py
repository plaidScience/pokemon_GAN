import tensorflow as tf
def get_avg_hue(img):
    avg_rgb = tf.reduce_mean(img, axis=[-3, -2])
    avg_hsv = tf.image.rgb_to_hsv(avg_rgb)
    avg_hue = avg_hsv[:, -3]
    return avg_hue

@tf.function
def tf_angDist(theta1, theta2, maxAng=1.0):
    return tf.math.minimum(tf.math.floormod(tf.math.subtract(theta1, theta2), maxAng), tf.math.floormod(tf.math.subtract(theta2, theta1), maxAng))