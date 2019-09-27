# Root directory for dataset
# DCGAN_IMAGE_ROOT = '/content/sample_data'
DCGAN_IMAGE_ROOT = 'D:\\pycharmspace\\cifar\\'
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