import os
from time import time
import datetime
import pickle
import tensorflow as tf

from unet import UNet
from utils import create_dataset, imshow, check_data
from constants import (
    LR, EPOCHS, BATCH_SIZE, NO_PROGRESS_EPOCHS,
    TRAIN_FRACTION, DATASET_DIRECTORY, RAW_IMAGES_PATH)
from train_test import train_step, test_step


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/' + current_time + '/train'
test_log_dir = 'logs/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


if __name__ == '__main__':
    try:
        # check if data folder exists and data is structured
        check_data()
        # create datasets if they do not exist
        if not os.path.exists('.data/train.pickle'):
            image_names = os.listdir(RAW_IMAGES_PATH)
            train_image_names = image_names[
                :int(len(image_names) * TRAIN_FRACTION)]
            val_image_names = image_names[
                int(len(image_names) * TRAIN_FRACTION):]
            create_dataset(
                file_names=train_image_names,
                dataset_name='train',
                batch_size=BATCH_SIZE)
            create_dataset(
                file_names=val_image_names,
                dataset_name='val',
                batch_size=BATCH_SIZE)

        train_dataset = pickle.load(
            open(DATASET_DIRECTORY + 'train.pickle', 'rb'))
        val_dataset = pickle.load(
            open(DATASET_DIRECTORY + 'val.pickle', 'rb'))
        train_crop_idx_list = train_dataset.get_batch_indices(_shuffle=True)
        val_crop_idx_list = val_dataset.get_batch_indices(_shuffle=False)

        loss_function = tf.keras.losses.MeanSquaredError()
        model = UNet()
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        val_loss = tf.keras.metrics.Mean(name='test_loss')
        model.compile(optimizer=optimizer, loss=loss_function)

        val_loss_list = []
        no_progress_counter = 0
        train_dataset = pickle.load(
            open(DATASET_DIRECTORY + 'train.pickle', 'rb'))
        val_dataset = pickle.load(
            open(DATASET_DIRECTORY + 'val.pickle', 'rb'))

        for epoch in range(EPOCHS):
            print('Start of epoch {}'.format(epoch))
            start_time = time()
            train_loss.reset_states()
            val_loss.reset_states()

            for i in range(train_dataset.__len__() // BATCH_SIZE):
                x_batch_train, y_batch_train = train_dataset.get_batch(
                    train_crop_idx_list, i)
                loss_value = train_step(
                    real=x_batch_train,
                    ground=y_batch_train,
                    model=model,
                    loss_function=loss_function,
                    optimizer=optimizer,
                    train_loss=train_loss)

            with train_summary_writer.as_default():
                tf.summary.scalar(
                    'Training loss',
                    train_loss.result(),
                    step=epoch)

            for i in range(val_dataset.__len__() // BATCH_SIZE):
                x_batch_val, y_batch_val = val_dataset.get_batch(
                    val_crop_idx_list, i)
                test_step(
                    real=x_batch_val,
                    ground=y_batch_val,
                    model=model,
                    loss_function=loss_function,
                    val_loss=val_loss)

            val_loss_list.append(val_loss.result())
            print(
                'Epoch {}, Loss: {:.5f}, Test Loss: {:.5f}, \
                    Time taken: {:.2f}s'.format(
                    epoch + 1,
                    train_loss.result(),
                    val_loss.result(),
                    time() - start_time))

            with test_summary_writer.as_default():
                tf.summary.scalar(
                    'Validation loss', val_loss.result(), step=epoch)
            with test_summary_writer.as_default():
                tf.summary.scalar(
                    'Time taken',
                    time() - start_time,
                    step=epoch)
            # check if training has reached a plateau
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
        x, real = val_dataset.get_batch(val_crop_idx_list, 48)
        fake = model(x[:1, :, :, :])
        imshow(real, fake)
        x, real = val_dataset.get_batch(val_crop_idx_list, 1138)
        fake = model(x[:1, :, :, :])
        imshow(real, fake)
