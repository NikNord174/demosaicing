import tensorflow as tf
from tf.keras import Model, models
from tensorflow.keras import Model


class UNet(Model):
    def __init__(self) -> None:
        super(UNet, self).__init__()

    def input(self, out_channels: int = 64,
              kernel_size: int = 3,) -> models.Sequential:
        layers = [
            tf.keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                activation='relu'),
            tf.keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                activation='relu'),
        ]
        return tf.keras.models.Sequential(layers)

    def down_block(self, out_channels: int = 128,
                   kernel_size: int = 3,) -> models.Sequential:
        layers = [
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                activation='relu'),
            tf.keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                activation='relu'),
        ]
        return tf.keras.models.Sequential(layers)

    def buttom(self, out_channels: int = 1024,
               kernel_size: int = 3,) -> models.Sequential:
        layers = [
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                activation='relu'),
            tf.keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                activation='relu'),
            tf.keras.layers.UpSampling2D(
                size=(2, 2),
                data_format='channels_last',
                interpolation='bilinear',),
            tf.keras.layers.Conv2D(
                filters=out_channels//2, kernel_size=1, activation='relu'),
        ]
        return tf.keras.models.Sequential(layers)

    def up_block(self,
                 x: tf.Tensor, y: tf.Tensor,
                 out_channels: int = 512,
                 kernel_size: int = 3,) -> models.Sequential:
        crop_size = (y.shape[-2]-x.shape[-2]) // 2
        cropped_y = tf.keras.layers.Cropping2D(cropping=crop_size)(y)
        x = tf.concat([x, cropped_y], -1)
        layers = [
            tf.keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                activation='relu'),
            tf.keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                activation='relu'),
            tf.keras.layers.UpSampling2D(
                size=(2, 2),
                data_format='channels_last',
                interpolation='bilinear',),
            tf.keras.layers.Conv2D(
                filters=out_channels/2, kernel_size=1, activation='relu'),
        ]
        return tf.keras.models.Sequential(layers)

    def output(self,
               x: tf.Tensor, y: tf.Tensor,
               out_channels: int = 64,
               kernel_size: int = 3,) -> models.Sequential:
        crop_size = (y.shape[-2]-x.shape[-2])//2
        cropped_y = tf.keras.layers.Cropping2D(cropping=crop_size)(y)
        x = tf.concat([x, cropped_y], -1)
        layers = [
            tf.keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                activation='relu'),
            tf.keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                activation='relu'),
            tf.keras.layers.Conv2D(
                filters=3, kernel_size=1, activation='sigmoid'),
        ]
        return tf.keras.models.Sequential(layers)

    def call(self, x):
        level_1 = self.input(out_channels=64)(x)
        level_2 = self.down_block(out_channels=128)(level_1)
        level_3 = self.down_block(out_channels=256)(level_2)
        level_4 = self.down_block(out_channels=512)(level_3)
        buttom = self.buttom(out_channels=1024)(level_4)
        level_up_1 = self.up_block(buttom, level_4, out_channels=512)(buttom)
        level_up_2 = self.up_block(
            level_up_1, level_3, out_channels=256)(level_up_1)
        level_up_3 = self.up_block(
            level_up_2, level_2, out_channels=128)(level_up_2)
        level_up_4 = self.output(
            level_up_3, level_1, out_channels=64)(level_up_3)
        return level_up_4


if __name__ == '__main__':
    test_image = tf.random.uniform(
        shape=[8, 572, 572, 3],
        minval=0,
        maxval=1,
        dtype=tf.dtypes.float32,
    )
    model = UNet()
    print(model(test_image).shape)
