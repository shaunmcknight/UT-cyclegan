import numpy as np
import itertools
import time
import datetime

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import make_grid
import torch.nn.functional as F

from torchinfo import summary

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from IPython.display import clear_output

# from PIL import Image
import PIL
import matplotlib.image as mpimg

from utils import *
from cyclegan import *

cuda = True if torch.cuda.is_available() else False
print("Using CUDA" if cuda else "Not using CUDA")

""" So generally both torch.Tensor and torch.cuda.Tensor are equivalent. You can do everything you like with them both.
The key difference is just that torch.Tensor occupies CPU memory while torch.cuda.Tensor occupies GPU memory.
Of course operations on a CPU Tensor are computed with CPU while operations for the GPU / CUDA Tensor are computed on GPU. """
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


##############################################
# Defining all hyperparameters
##############################################

class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


hp = Hyperparameters(
    epoch=0,
    n_epochs=200,
    dataset_train_mode="train",
    dataset_test_mode="test",
    batch_size=4,
    lr=0.0002,
    decay_start_epoch=100,
    b1=0.5,
    b2=0.999,
    n_cpu=8,
    img_size=128,
    channels=3,
    n_critic=5,
    sample_interval=400,
    num_residual_blocks=19, #changed from 19
    lambda_cyc=10.0,
    lambda_id=5.0,
)

root_path = "/home/euan/Shaun/Datasets/summer2winter_yosemite"


########################################################
# Methods for Image Visualization
########################################################
def show_img(img, size=10):
    npimg = img.cpu().detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.figure(figsize=(size, size))
    plt.show()


def to_img(x):
    x = x.view(x.size(0) * 2, hp.channels, hp.img_size, hp.img_size)
    return x


def plot_output(path, x, y):
    img = mpimg.imread(path)
    plt.figure(figsize=(x, y))
    plt.imshow(img)
    plt.show()


##############################################
# Defining Image Transforms to apply
##############################################
transforms_ = [
    transforms.Resize((hp.img_size, hp.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


##############################################
# Initialize generator and discriminator
##############################################

input_shape = (hp.channels, hp.img_size, hp.img_size)

Gen_AB = GeneratorResNet(input_shape, hp.num_residual_blocks)
Gen_BA = GeneratorResNet(input_shape, hp.num_residual_blocks)


if cuda:
    Gen_AB = Gen_AB.cuda()
    Gen_BA = Gen_BA.cuda()

##############################################
# Load weights
##############################################

Gen_AB.apply(initialize_conv_weights_normal)
Gen_AB.load_state_dict(torch.load('saved_Gen_AB.pt'))

Gen_BA.apply(initialize_conv_weights_normal)
Gen_BA.load_state_dict(torch.load('saved_Gen_BA.pt'))

##############################################
# Final Validation Function
##############################################


def validate(
        Gen_AB,
        Gen_BA,
            
):
    
    start_time = time.time()
    i = 55
    val_data = ImageDataset(root_path, mode=hp.dataset_test_mode, transforms_=transforms_)
    batch = val_data[i]

    # Set model input
    real_A = Variable(batch["A"].type(Tensor))
    real_B = Variable(batch["B"].type(Tensor))

    Gen_AB.eval()
    Gen_BA.eval()

    fake_B = Gen_AB(real_A.unsqueeze(0))
    fake_A = Gen_BA(real_B.unsqueeze(0))
    
    reconstructed_A = Gen_BA(fake_B)
    reconstructed_B = Gen_AB(fake_A)
    
    undo_normalisation = transforms.Compose([
        transforms.Normalize((-0.5/0.5, -0.5/0.5, -0.5/0.5), (1/0.5, 1/0.5, 1/0.5))
        ])

    real_A = undo_normalisation(real_A.cpu())
    fake_B = undo_normalisation(fake_B.squeeze(0).cpu().detach())
    reconstructed_A = undo_normalisation(reconstructed_A.squeeze(0).cpu().detach())
    
    real_B = undo_normalisation(real_B.cpu())
    fake_A = undo_normalisation(fake_A.squeeze(0).cpu().detach())
    reconstructed_B = undo_normalisation(reconstructed_B.squeeze(0).cpu().detach())

    show_img(real_A, 30)
    show_img(fake_B, 30)
    show_img(reconstructed_A, 30)
    
    show_img(real_B, 30)    
    show_img(fake_A, 30)
    show_img(reconstructed_B, 30)    
    
    image_grid = make_grid([real_A, fake_B, reconstructed_A, real_B, fake_A, reconstructed_B], normalize=True)
    show_img(image_grid, 30)
    
    path = "/home/euan/Shaun/Datasets/summer2winter_yosemite/validate%s.png" % (i)
    save_image(image_grid, path, normalize=False)
    
    # print(summary(Gen_BA, input_size=(3,128,128)))
    print("Total time taken ", str(time.time()-start_time))



##############################################
# Execute the Final Training Function
##############################################

validate(
    Gen_BA=Gen_BA,
    Gen_AB=Gen_AB,
)