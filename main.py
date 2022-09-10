import os
from time import time
import datetime
import tensorflow as tf

from unet import UNet
# from utils import check_data  # <------ uncomment
from constants import (
    LR, EPOCHS, NO_PROGRESS_EPOCHS, TRAIN_FRACTION, TEST_FRACTION)
from train_test import train_step, test_step
from dataset import Image_Dataset
from losses import MSE


RGB_IMAGES_PATH = 'data/rgb_images'
RAW_IMAGES_PATH = 'data/raw_images'


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


train_dataset = Image_Dataset()
val_dataset = Image_Dataset(train=False)
train_loss = MSE
val_loss = MSE
model = UNet()
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)


if __name__ == '__main__':
    try:
        # check_data(RAW_IMAGES_PATH, RGB_IMAGES_PATH)  # <------ uncomment
        val_loss_list = []
        no_progress_counter = 0
        for epoch in range(EPOCHS):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time()
            for idx in range(int(len(os.listdir(RAW_IMAGES_PATH)) * TRAIN_FRACTION)):
                x_batch_train, y_batch_train = train_dataset.slice_image(idx)
                loss_value = train_step(
                    x_batch_train, y_batch_train, model, optimizer)
                with train_summary_writer.as_default():
                    tf.summary.scalar(
                        'batch_loss', loss_value, step=epoch)
            train_loss_epoch = train_loss.result()
            print("Training loss over epoch: %.4f" % (float(train_loss_epoch),))
            with train_summary_writer.as_default():
                tf.summary.scalar(
                    'loss', train_loss.result(), step=epoch)

            for idx in range(int(len(os.listdir(RAW_IMAGES_PATH)) * TEST_FRACTION)):
                x_batch_val, y_batch_val = val_dataset.slice_image(idx)
                test_step(x_batch_val, y_batch_val, model)
            val_loss_epoch = val_loss.result()
            val_loss_list.append(val_loss_epoch)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', val_loss.result(), step=epoch)
            print("Validation loss: %.4f" % (float(val_loss_epoch),))
            print("Time taken: %.2fs" % (time() - start_time))
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

    '''for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train, y_batch_train)

    for x_batch_val, y_batch_val in val_dataset:
        test_step(x_batch_val, y_batch_val)'''
