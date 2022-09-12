import os
import random
import colour
import tensorflow as tf


class Image_Dataset():
    def __init__(
        self,
        file_names: list,
        crop_height: int = 128,
        crop_width: int = 128,
        batch_size: int = 64,
        image_height: int = 4032,
        image_width: int = 3024
    ) -> None:
        self.image_names = [name.split('.')[0] for name in file_names]
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.crops_coordinates = self.get_crops_coordinates()
        self.raw_images, self.rgb_images = self.load_images()

    def load_images(self):
        raw_images_initial = [tf.transpose(
            tf.convert_to_tensor(
                colour.io.read_image(
                    os.path.join(
                        f'data/raw_images/{raw_image}.dng'))
                ), [1, 0]) for raw_image in self.image_names]
        raw_images = [tf.reshape(
            raw_image, [raw_image.shape[0], raw_image.shape[1], 1])
            for raw_image in raw_images_initial]
        rgb_images = [tf.transpose(tf.io.decode_image(
            tf.io.read_file(
                f'data/rgb_images/{rgb_image}.png'),
            channels=3,
            dtype=tf.dtypes.float32), [1, 0, 2])
            for rgb_image in self.image_names]
        return raw_images, rgb_images

    def overlap(self, image_side: int, crop_side: int) -> int:
        """Calculate overlap depends on crop and image sizes"""
        crops_num = image_side // crop_side
        extra_pixels = image_side - crops_num * crop_side
        return (crop_side - extra_pixels) // (crops_num + 1)

    def get_crops_coordinates(self) -> list[list[int]]:
        """Calculate list of lists with 4 coordinates of crops"""
        crop_coordinates = []
        if not self.image_height % self.crop_height == 0:
            overlap_height = self.overlap(
                image_side=self.image_height,
                crop_side=self.crop_height)
        if not self.image_width % self.crop_width == 0:
            overlap_width = self.overlap(
                image_side=self.image_width,
                crop_side=self.crop_width)
        y_min = 0
        y_max = y_min + self.crop_height
        while y_max < self.image_height:
            x_min = 0
            x_max = x_min + self.crop_width
            while x_max < self.image_width:
                crop_coordinates.append([x_min, y_min, x_max, y_max])
                x_min = x_max - overlap_width
                x_max = x_min + self.crop_width
            y_min = y_max - overlap_height
            y_max = y_min + self.crop_height
        return crop_coordinates

    def __len__(self) -> int:
        """Returns number of sclices in dataset"""
        return len(self.image_names) * len(self.crops_coordinates)

    def crop_image(self, idx: int) -> tf.Tensor:
        """Returns certain crop from certain image"""
        image_idx = idx // len(self.crops_coordinates)
        crop_idx = idx % len(self.crops_coordinates)
        raw_image = self.raw_images[image_idx]
        rgb_image = self.rgb_images[image_idx]
        coord = self.crops_coordinates[crop_idx]
        sample = tf.convert_to_tensor(
            raw_image[coord[1]:coord[3], coord[0]:coord[2], :])
        target = tf.convert_to_tensor(
            rgb_image[coord[1]:coord[3], coord[0]:coord[2], :])
        return sample, target

    def get_batch_indices(self, _shuffle: bool) -> list:
        """Returns list of indeces for batch"""
        idxs = list(range(self.__len__()))
        if not _shuffle:
            return idxs
        random.shuffle(idxs)
        return idxs

    def get_batch(
        self,
        crop_idx_list: list,
        counter: int
    ) -> list[tf.Tensor, tf.Tensor]:
        """Returns batch of crops"""
        start = counter
        if not (counter + self.batch_size >= len(crop_idx_list)):
            stop = counter + self.batch_size
            sample = tf.convert_to_tensor(
                [self.crop_image(idx)[0] for idx in crop_idx_list[start:stop]])
            target = tf.convert_to_tensor(
                [self.crop_image(idx)[1] for idx in crop_idx_list[start:stop]])
            return sample, target
        sample = tf.convert_to_tensor(
            [self.crop_image(idx)[0] for idx in crop_idx_list[start:]])
        target = tf.convert_to_tensor(
            [self.crop_image(idx)[1] for idx in crop_idx_list[start:]])
        return sample, target
