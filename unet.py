import tensorflow as tf

from constants import POOL_FACTOR, POOL_MODE, ReLU_FACTOR


class UNet(tf.keras.Model):
    def __init__(self) -> None:
        super(UNet, self).__init__()
        self._level_1 = self.input(out_channels=64)
        self._level_2 = self.down_block(out_channels=128)
        self._bottom = self.bottom(out_channels=512)
        self._level_up_1 = self.up_block(out_channels=64)
        self._level_up_2 = self.output(out_channels=64)

    def conv_block(
        self,
        filters: int = 64,
        padding: str = 'same',
        kernel_size: int = 3,
    ) -> tf.keras.models.Sequential:
        layers = [
            tf.keras.layers.Conv2D(
                filters=filters,
                padding=padding,
                kernel_size=kernel_size,
                activation=None),
            tf.keras.layers.BatchNormalization(axis=-1),
            tf.keras.layers.LeakyReLU(alpha=ReLU_FACTOR),
        ]
        return tf.keras.models.Sequential(layers)

    def input(
        self,
        out_channels: int = 64,
        kernel_size: int = 3,
    ) -> tf.keras.models.Sequential:
        layers = [
            self.conv_block(filters=out_channels, kernel_size=kernel_size),
            self.conv_block(filters=out_channels, kernel_size=kernel_size),
        ]
        return tf.keras.models.Sequential(layers)

    def down_block(
        self,
        out_channels: int = 128,
        kernel_size: int = 3,
    ) -> tf.keras.models.Sequential:
        layers = [
            tf.keras.layers.MaxPool2D(pool_size=(POOL_FACTOR, POOL_FACTOR)),
            self.conv_block(filters=out_channels, kernel_size=kernel_size),
            self.conv_block(filters=out_channels, kernel_size=kernel_size),
        ]
        return tf.keras.models.Sequential(layers)

    def bottom(
        self,
        out_channels: int = 1024,
        kernel_size: int = 3,
    ) -> tf.keras.models.Sequential:
        layers = [
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            self.conv_block(filters=out_channels, kernel_size=kernel_size),
            self.conv_block(filters=out_channels, kernel_size=kernel_size),
            tf.keras.layers.UpSampling2D(
                size=(POOL_FACTOR, POOL_FACTOR),
                data_format='channels_last',
                interpolation=POOL_MODE,),
            self.conv_block(filters=out_channels//2, kernel_size=1),
        ]
        return tf.keras.models.Sequential(layers)

    def crop(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        crop_size = (y.shape[-2]-x.shape[-2]) // 2
        if y.shape[-2] % 2 != 0:
            cropped_y = tf.keras.layers.Cropping2D(
                cropping=(
                    (crop_size, crop_size+1), (crop_size, crop_size+1)))(y)
        else:
            cropped_y = tf.keras.layers.Cropping2D(cropping=crop_size)(y)
        return tf.concat([x, cropped_y], -1)

    def up_block(
        self,
        out_channels: int = 512,
        kernel_size: int = 3,
    ) -> tf.keras.models.Sequential:
        layers = [
            self.conv_block(filters=out_channels, kernel_size=kernel_size),
            self.conv_block(filters=out_channels, kernel_size=kernel_size),
            tf.keras.layers.UpSampling2D(
                size=(POOL_FACTOR, POOL_FACTOR),
                data_format='channels_last',
                interpolation=POOL_MODE,),
            self.conv_block(filters=out_channels//2, kernel_size=1),
        ]
        return tf.keras.models.Sequential(layers)

    def output(
        self,
        out_channels: int = 64,
        kernel_size: int = 3,
    ) -> tf.keras.models.Sequential:
        layers = [
            self.conv_block(filters=out_channels, kernel_size=kernel_size),
            self.conv_block(filters=out_channels, kernel_size=kernel_size),
            tf.keras.layers.Conv2D(
                filters=3,
                kernel_size=1,
                activation='sigmoid'),
        ]
        return tf.keras.models.Sequential(layers)

    def call(self, x):
        level_1 = self._level_1(x)
        level_2 = self._level_2(level_1)
        bottom = self._bottom(level_2)
        x = self.crop(bottom, level_2)
        level_up_1 = self._level_up_1(x)
        x = self.crop(level_up_1, level_1)
        level_up_2 = self._level_up_2(x)
        return level_up_2
