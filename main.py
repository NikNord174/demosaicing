import logging
from time import time
from datetime import datetime
from logging.handlers import RotatingFileHandler

from unet import UNet
from constants import (
    LR, BATCH_SIZE, EPOCHS, NO_PROGRESS_EPOCHS)


logging.basicConfig(
    level=logging.INFO,
    filename='results/Experiments.log',
    filemode='a',
    format='%(message)s'
)

# logging into file
file = logging.getLogger(__name__)
file.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(message)s')
file_handler = RotatingFileHandler(
    'results/Experiments_main.log',
    mode='a', maxBytes=5*1024*1024,
    backupCount=2)
file_handler.setFormatter(file_formatter)
file.addHandler(file_handler)

# console output
console = logging.getLogger('console')
console.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_formatter)
console.addHandler(console_handler)


model = UNet()

if __name__ == '__main__':
    try:
        date = datetime.now()
        file.info('Experiment: {}'.format(
            date.strftime('%m/%d/%Y, %H:%M:%S')))
        file.info('Model: {}'.format(model.__class__.__name__))
        file.info('Model detail: {}'.format(model.__repr__()))
        #file.info(f'Loss: {MSE_COEF}*MSE+{SSIM_COEF}*SSIM')
        file.info('Batch size: {}'.format(BATCH_SIZE))
        file.info('Learning rate: {}'.format(LR))
        comment = input('Comment: ')
        file.info('Comment: {}'.format(comment))
        t0 = time()
        test_loss_list = []
        n = 0
        for epoch in range(EPOCHS):
            



            
            t1 = (time() - t0) / 60
            msg = 'Epoch: {}, test loss: {:.5f}, time: {:.2f} min'.format(
                    epoch+1, test_loss, t1)
            file.info(msg)
            console.info(msg)
            if test_loss <= min(test_loss_list):
                n = 0
                continue
            else:
                n += 1
                if n > NO_PROGRESS_EPOCHS:
                    progress_msg = 'No progress for more than 5 epochs'
                    file.info(progress_msg)
                    console.info(progress_msg)
                    break
    except KeyboardInterrupt:
        raise KeyboardInterrupt('Learning has been stopped manually')
    finally:
        file.info('----------------')
