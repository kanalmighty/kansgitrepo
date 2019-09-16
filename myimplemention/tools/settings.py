TEST_IMAGE_ROOT_PATH = 'D:\\pycharmspace\\testimage\\'
TEST_RAW_IMAGE_PATH = TEST_IMAGE_ROOT_PATH + 'raw\\'
TEST_GEOMETRY_IMAGE_PATH = TEST_IMAGE_ROOT_PATH + 'rotated\\'

PASSWORD = '123456'

POOL_LOWER_THRESHOLD = 10
POOL_UPPER_THRESHOLD = 100

CROPED_WIDTH = 511
CROPED_HEIGHT = 383

VALID_CHECK_CYCLE = 60
POOL_LEN_CHECK_CYCLE = 20

TEST_API = 'https://www.baidu.com'

# Root directory for dataset
DCGAN_IMAGE_ROOT = '/content/sample_data'

# Number of workers for dataloader
WORKERS = 0

# Batch size during training
BATCH_SIZE = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
IMAGE_SIZE = 64

# Number of channels in the training images. For color images this is 3
NC = 3

# Size of z latent vector (i.e. size of generator input)
NZ = 100

# Size of feature maps in generator
NGF = 64

# Size of feature maps in discriminator
NDF = 64

# Number of training epochs
NUM_EPOCHS = 5

# Learning rate for optimizers
LR = 0.0002

# Beta1 hyperparam for Adam optimizers
BETA1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
NGPU = 0