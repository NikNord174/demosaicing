# models parameters
POOL_MODE = 'bilinear'
ReLU_FACTOR = 0.2

# learning parameters
BATCH_SIZE = 64
NUM_WORKERS = 2
EPOCHS = 30
LR = 1e-4
NO_PROGRESS_EPOCHS = 5

# coefficients for ssim and mse in learning process
SSIM_COEF = 0.60
MSE_COEF = 1 - SSIM_COEF
