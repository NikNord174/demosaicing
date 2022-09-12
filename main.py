import os
from time import time
import datetime
import tensorflow as tf

from unet import UNet
from utils import check_data
from constants import (
    LR, EPOCHS, BATCH_SIZE, NO_PROGRESS_EPOCHS, TRAIN_FRACTION)
from train_test import train_step, test_step
from dataset import Image_Dataset
from losses import MSE
from utils import imshow


RGB_IMAGES_PATH = 'data/rgb_images'
RAW_IMAGES_PATH = 'data/raw_images'


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


image_names = os.listdir(RAW_IMAGES_PATH)
train_image_names = image_names[:int(len(image_names) * TRAIN_FRACTION)]
val_image_names = image_names[int(len(image_names) * TRAIN_FRACTION):]
train_dataset = Image_Dataset(
    file_names=train_image_names, batch_size=BATCH_SIZE)
val_dataset = Image_Dataset(
    file_names=val_image_names, batch_size=BATCH_SIZE)
train_loss = MSE
val_loss = MSE
model = UNet()
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)


if __name__ == '__main__':
    try:
        check_data(RAW_IMAGES_PATH, RGB_IMAGES_PATH)
        val_loss_list = []
        no_progress_counter = 0
        for epoch in range(EPOCHS):
            print('Start of epoch {}'.format(epoch))
            start_time = time()

            crop_idx_list = train_dataset.get_batch_indices(_shuffle=True)
            for i in range(train_dataset.__len__() // BATCH_SIZE):
                x_batch_train, y_batch_train = train_dataset.get_batch(
                    crop_idx_list, i)
                loss_value = train_step(
                    x_batch_train, y_batch_train, model, optimizer)
            train_loss_epoch = train_loss.result()
            print('Training loss over epoch: {:.4f}'.format(
                float(train_loss_epoch),))
            with train_summary_writer.as_default():
                tf.summary.scalar(
                    'Training loss over epoch',
                    train_loss.result(),
                    step=epoch)

            crop_idx_list = val_dataset.get_batch_indices(_shuffle=False)
            for i in range(val_dataset.__len__() // BATCH_SIZE):
                x_batch_val, y_batch_val = val_dataset.get_batch(
                    crop_idx_list, i)
                test_step(x_batch_val, y_batch_val, model)
            val_loss_epoch = val_loss.result()
            val_loss_list.append(val_loss_epoch)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', val_loss.result(), step=epoch)
            print('Validation loss: {:.4f}'.format(float(val_loss_epoch),))
            with test_summary_writer.as_default():
                tf.summary.scalar(
                    'Validation loss',
                    val_loss.result(),
                    step=epoch)
            print('Time taken: {:.2f}s'.format(time() - start_time))
            with test_summary_writer.as_default():
                tf.summary.scalar(
                    'Time taken',
                    time() - start_time,
                    step=epoch)
            train_loss.reset_states()
            val_loss.reset_states()

            if val_loss_list[-1] <= min(val_loss_list):
                no_progress_counter = 0
                continue
            else:
                no_progress_counter += 1
                if no_progress_counter > NO_PROGRESS_EPOCHS:
                    print('No progress for more than 5 epochs')
                    break
    except KeyboardInterrupt:
        raise KeyboardInterrupt('Learning has been stopped manually')
    finally:
        x, real = val_dataset.get_batch(crop_idx_list, 12)
        fake = model(x[:1, :, :, :])
        imshow(real, fake)
