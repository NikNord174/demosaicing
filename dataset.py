import os
import colour
import tensorflow as tf
from constants import TRAIN_FRACTION, TEST_FRACTION


class Image_Dataset():
    def __init__(
        self,
        raw_image_path: str = 'data/raw_images',
        train: bool = True,
        slice_height: int = 128,
        slice_width: int = 128,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
    ) -> None:
        files = os.listdir(raw_image_path)
        files.remove('.DS_Store')  # <------ delete
        if train:
            self.image_names = [
                name.split('.')[0]
                for name in files[:int(len(files) * TRAIN_FRACTION)]]
        else:
            self.image_names = [
                name.split('.')[0]
                for name in files[:int(len(files) * TEST_FRACTION)]]
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio

    def __len__(self):
        return len(self.image_names)

    def get_slice_coordinates(
        self, image_height: int = 3024,
        image_width: int = 4032
    ) -> list[list[int]]:
        slice_bboxes = []
        y_min = 0
        y_max = y_min + self.slice_height
        while y_max < image_height:
            x_min = 0
            x_max = x_min + self.slice_width
            while x_max < image_width:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
                x_min = x_max
                x_max = x_min + self.slice_width
            y_min = y_max
            y_max = y_min + self.slice_height
        return slice_bboxes

    def slice_image(self, idx: int) -> tf.Tensor:
        raw_image = tf.transpose(
            tf.convert_to_tensor(
                colour.io.read_image(
                    os.path.join(
                        f'data/raw_images/{self.image_names[idx]}.dng'))
                ), [1, 0])
        raw_image = tf.reshape(
            raw_image, [raw_image.shape[0], raw_image.shape[1], 1])
        rgb_image = tf.transpose(tf.io.decode_image(
            tf.io.read_file(
                f'data/rgb_images/{self.image_names[idx]}.png'),
            channels=3,
            dtype=tf.dtypes.float32), [1, 0, 2])
        slices_coordinates = self.get_slice_coordinates(
            image_height=raw_image.shape[0],
            image_width=raw_image.shape[1])
        samples = tf.convert_to_tensor(
            [raw_image[coord[1]:coord[3], coord[0]:coord[2], :]
                for coord in slices_coordinates])
        targets = tf.convert_to_tensor(
            [rgb_image[coord[1]:coord[3], coord[0]:coord[2], :]
                for coord in slices_coordinates])
        return samples[:128, :, :, :], targets[:128, :, :, :]
