# dataset parameters
DATASET_DIRECTORY: str = 'data/'
DATASET_NAME: str = 'train'
TRAIN_FRACTION: float = 0.75
CROP_HEIGHT: int = 128
CROP_WIDTH: int = 128

# models parameters
POOL_MODE: str = 'bilinear'
ReLU_FACTOR: float = 0.2

# learning parameters
BATCH_SIZE: int = 64
EPOCHS: int = 20
LR: float = 1e-3
NO_PROGRESS_EPOCHS: int = 5
