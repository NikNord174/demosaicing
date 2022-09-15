# dataset parameters
DATASET_DIRECTORY: str = 'data/'
RGB_IMAGES_PATH: str = 'data/rgb_images'
RAW_IMAGES_PATH: str = 'data/raw_images'
DATASET_NAME: str = 'train'
TRAIN_FRACTION: float = 0.85
CROP_HEIGHT: int = 128
CROP_WIDTH: int = 128

# models parameters
POOL_MODE: str = 'bilinear'
POOL_FACTOR: int = 2
ReLU_FACTOR: float = 0.2

# learning parameters
BATCH_SIZE: int = 64
EPOCHS: int = 20
LR: float = 1e-3
NO_PROGRESS_EPOCHS: int = 5
