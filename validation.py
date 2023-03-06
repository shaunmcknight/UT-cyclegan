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
import scienceplots
from mpl_toolkits.axes_grid1 import ImageGrid
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

plt.style.use(['science', 'ieee','no-latex', 'bright'])

class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
class CIVADataset(Dataset):
    def __init__(self, transforms_=None):
        
        self.root = r"C:/Users/CUE-ML/shaun/Datasets/CIVA/3_6_9_dataset_0_1_norm.npy"
        # self.root = "/media/cue-server/New Volume/ShaunMcKnight/Datasets/CIVA/3_6_9_dataset_augmented_test.npy"
        self.dataset = np.load(self.root)

        self.mean = self.dataset.mean()
        self.std = self.dataset.std()
        
        self.transform = None #transforms.Normalize((self.mean), (self.std))

    def __getitem__(self, index):
        image = torch.from_numpy(self.dataset[index])
        image = image.unsqueeze(0)

        if self.transform != None:
            image = self.transform(image)
                
        return image

    def __len__(self):
        return len(self.dataset)
    
hp = Hyperparameters(
    epoch=0,
    n_epochs=3000,
    dataset_train_mode="train",
    dataset_test_mode="test",
    batch_size=128,
    lr=0.0002,
    decay_start_epoch=100,
    b1=0.5,
    b2=0.999,
    n_cpu=8,
    img_size=64,
    channels=1,
    n_critic=5,
    sample_interval=50,
    num_residual_blocks=6,
    lambda_cyc=100,
    lambda_id=0,#5.0
    lambda_mid = 200
)

# model_iter = "iter36/"
model_iter = ""#"iter37/"
model_epoch = "_6850"
# model_epoch = ""

if model_iter  != None:
    model_path = r"C:/Users/CUE-ML/shaun/Python/UT_CycleGAN/" + model_iter
else:
    model_path = r"C:/Users/CUE-ML/shaun/Python/UT_CycleGAN"
    
root_path = r"C:/Users/CUE-ML/shaun/Datasets/civa2experimental"

########################################################
# Methods for Image Visualization
########################################################
def show_img(img, size=10):
    npimg = img.cpu().detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)).squeeze())# , cmap = 'inferno')
    plt.colorbar()
    plt.figure(figsize=(size, size))
    plt.show()

def show_grid(im1, im2, im3, im4, im5, im6):
    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(2, 3),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

    for ax, im in zip(grid, [im1, im2, im3, im4, im5, im6]):
        # Iterating over the grid returns the Axes.
        ax.imshow(np.transpose(im, (1, 2, 0)).squeeze())
    
    plt.show()
    
def plotAllImages(synthetic_dataset, civa_dataset):
    for i in range(0, np.shape(synthetic_dataset)[0]):
        
        plt.figure(figsize = (5,2))
        plt.subplot(1,2,1)
        plt.imshow(civa_dataset[i])
        # plt.colorbar()
        # plt.axis('off')

        # plt.colorbar()
        plt.subplot(1,2,2)
        plt.imshow(synthetic_dataset[i])
        # plt.colorbar()
        # plt.axis('off')
        # plt.show()
        plt.show()
    
def plotAllImagesDb(synthetic_dataset, civa_dataset):
    for i in range(0, np.shape(synthetic_dataset)[0]):
        
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(civa_dataset[i])
        # plt.colorbar()
        # plt.axis('off')

        # plt.colorbar()
        plt.subplot(2,2,2)
        plt.imshow(synthetic_dataset[i])
        # plt.colorbar()
        # plt.axis('off')
        # plt.show()
        
        plt.subplot(2,2,3)
        plt.imshow(20*np.log10(civa_dataset[i]/np.max(civa_dataset[i])))
        plt.clim(0, -6)
        # plt.colorbar()
        # plt.axis('off')

        db_synthetic = synthetic_dataset[i]-np.min(synthetic_dataset[i])
        db_synthetic = db_synthetic/np.nanmax(db_synthetic)
        # plt.colorbar()
        plt.subplot(2,2,4)
        plt.imshow(20*np.log(abs(db_synthetic)))
        plt.clim(0, -6)
        # plt.colorbar()
        # plt.axis('off')
        
        plt.show()
        
# def single_img(img, size=10):
    
#     i = 9
#     val_data = ImageDataset(root_path, mode=hp.dataset_test_mode)
#     batch = val_data[i]

#     # Set model input
#     img = Variable(batch["A"].type(Tensor))

#     npimg = img.cpu().detach().numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.figure()
#     plt.show()


def to_img(x):
    x = x.view(x.size(0) * 2, hp.channels, hp.img_size, hp.img_size)
    return x


def plot_output(path, x, y):
    img = mpimg.imread(path)
    plt.figure(figsize=(x, y))
    plt.imshow(img)
    plt.show()

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
print(summary(Gen_BA, input_size=(1,64,64)))

Gen_AB.load_state_dict(torch.load(model_iter + 'saved_Gen_AB' + model_epoch + '.pt')) #saved_Gen_AB_37600

Gen_BA.apply(initialize_conv_weights_normal)
Gen_BA.load_state_dict(torch.load(model_iter + 'saved_Gen_BA' + model_epoch + '.pt'))

##############################################
# Final Validation Function
##############################################
    

def validate(
        Gen_AB,
        Gen_BA,
    ):
    
    start_time = time.time()
    i = 10
    for i in range(i):
        val_data = ImageDataset(root_path, mode=hp.dataset_test_mode, transforms_ = None)
        batch = val_data[i]
    
        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))
        
        random = np.random.random_sample(real_A.shape)*0.0005
        random = torch.from_numpy(random).type(Tensor)
        random = 0
     
        real_A_rand = real_A+random
    
        Gen_AB.eval()
        Gen_BA.eval()
    
        fake_B = Gen_AB(real_A.unsqueeze(0))
        fake_A = Gen_BA(real_B.unsqueeze(0))
        
        reconstructed_A = Gen_BA(fake_B)
        reconstructed_B = Gen_AB(fake_A)
        
        undo_normalisation = transforms.Compose([
            # transforms.Normalize((-0.5/0.5), (1/0.5))
            ])

        real_A = undo_normalisation(real_A.cpu())
        
        fake_B = undo_normalisation(fake_B.squeeze(0).cpu().detach())
        reconstructed_A = undo_normalisation(reconstructed_A.squeeze(0).cpu().detach())
        
        real_B = undo_normalisation(real_B.cpu())
        fake_A = undo_normalisation(fake_A.squeeze(0).cpu().detach())
        reconstructed_B = undo_normalisation(reconstructed_B.squeeze(0).cpu().detach())
    
        # show_img(real_A, 30)
        # show_img(fake_B, 30)
        # show_img(reconstructed_A, 30)
        
        # show_img(real_B, 30)    
        # show_img(fake_A, 30)
        # show_img(reconstructed_B, 30)    
        
        show_grid(real_A, fake_B, reconstructed_A, real_B, fake_A, reconstructed_B)
        
        # fig = plt.figure(figsize=(4., 4.))
        # grid = ImageGrid(fig, 111,  # similar to subplot(111)
        #              nrows_ncols=(2, 3),  # creates 2x2 grid of axes
        #              axes_pad=0.1,  # pad between axes in inch.
        #              )

        # for ax, im in zip(grid, [real_A, fake_B, reconstructed_A]):#, real_B, fake_A, reconstructed_B]):
        #     # Iterating over the grid returns the Axes.
        #     ax.imshow(np.transpose(im, (1, 2, 0)).squeeze())
            
        # plt.show()
        
        
        # image_grid = make_grid([real_A, fake_B, reconstructed_A, real_B, fake_A, reconstructed_B], normalize=True)
        # show_img(image_grid, 30)
        
        # path = "/home/euan/Shaun/Datasets/summer2winter_yosemite/validate%s.png" % (i)
        # save_image(image_grid, path, normalize=False)
        
        # print(summary(Gen_BA, input_size=(3,128,128)))
        print("Total time taken ", str(time.time()-start_time))

def plot_losses(data_path):
    losses = np.load(model_path+r'/losses.npz')
    loss_D = losses['D'] #Discriminator losses
    loss_D_AB = losses['D_AB'] #Discriminator losses
    loss_D_BA = losses['D_BA'] #Discriminator losses
    loss_G_AB = losses['G_AB'] #Generator losses
    loss_G_BA = losses['G_BA'] #Generator losses
    loss_G = losses['G'] #Generator losses
    loss_cycle = losses['cycle'] 
    loss_identity = losses['identity']
    loss_mid = losses['mid']
    loss_GAN = losses['GAN'] # complete GAN losses combined

    
    plt.figure()
    plt.plot(loss_D, label = 'Total (mean) loss')
    plt.plot(loss_D_AB, label = 'AB loss')
    plt.plot(loss_D_BA, label = 'BA loss')
    plt.title('Discriminator Loss')
    plt.legend()
    plt.show()
    
    # plt.figure()
    # plt.plot(loss_D[150:])
    # plt.title('Discriminator Loss')
    # plt.show()
    
    # plt.figure()
    # plt.plot(loss_G)
    # plt.title('Generator Loss')
    # plt.show()

    # plt.figure()
    # plt.plot(loss_G_AB)
    # plt.title('Generator Loss AB')
    # plt.show()

    # plt.figure()
    # plt.plot(loss_G_BA)
    # plt.title('Generator Loss BA')
    # plt.show()
    
    # plt.figure()
    # plt.plot(loss_cycle)
    # plt.title('Cycle Loss')
    # plt.show()
    
    # plt.figure()
    # plt.plot(loss_identity)
    # plt.title('Identity Loss')
    # plt.show()
    
    # plt.figure()
    # plt.plot(loss_GAN)
    # plt.title('GAN Loss')
    # plt.show()
    
    # plt.figure()
    # plt.plot(loss_mid)
    # plt.title('Mid cycle Loss')
    # plt.show()
    
    plt.figure()
    plt.plot(loss_G, label = 'Total Generator loss')
    plt.plot(loss_cycle, label = 'Cycle loss')
    plt.plot(loss_mid, label = 'Mid cycle loss')
    plt.plot((loss_G_AB+loss_G_BA)/2, label = 'Generator loss')
    plt.legend(loc="upper left")
    plt.title('Generator loss')
    plt.show()
        
    plt.figure()
    plt.plot(loss_G, label = 'Total Generator loss')
    plt.plot(loss_cycle*hp.lambda_cyc, label = 'Cycle loss')
    plt.plot(loss_mid*hp.lambda_mid, label = 'Mid cycle loss')
    plt.plot((loss_G_AB+loss_G_BA)/2, label = 'Generator loss')
    plt.legend(loc="upper left")
    plt.title('Generator loss scaled')
    plt.ylim([0, 800])
    plt.show()
    
    # plt.figure()
    # plt.plot(loss_G, loss_GAN, loss_cycle, loss_identity)
    # plt.legend()
    # plt.show()
    
    
def generate_synthetic_data(
        Gen_AB,
        Gen_BA,
    ):
    
    synthetic_dataset = []
    civa_dataset = []
    # i = 10
    for i in range(len(CIVADataset(transforms_=None))):
        val_data = CIVADataset(transforms_ = None)
        # print(np.shape(val_data))
        CIVA_data = val_data[i].type(Tensor)
        
        # random = np.random.random_sample(CIVA_data.shape)*0.0005
        # random = torch.from_numpy(random).type(Tensor)
        random = 0
        
        # civa_dataset.append((CIVA_data+random).cpu().detach().squeeze())
        
        Gen_AB.eval()
        Gen_BA.eval()
    
        # synth_data = Gen_AB((CIVA_data+random).unsqueeze(0))
                
        # transform = transforms.Compose([
        #     transforms.RandomAffine(degrees = 0, translate=(0.1, 0.1))
        #     ])
        
        # CIVA_data = transform(CIVA_data)
        
        civa_dataset.append((CIVA_data).cpu().detach().squeeze())

        synth_data = Gen_AB((CIVA_data).unsqueeze(0))

        
        # synth_data = undo_normalisation(synth_data.squeeze(0).cpu().detach())
        
        # plt.figure()
        # plt.imshow(CIVA_data.cpu().squeeze())
        
        synthetic_dataset.append(synth_data.cpu().detach().squeeze())
        
        # plt.figure()
        # plt.imshow(synth_data.cpu().detach().squeeze())
        # plt.colorbar()
    
    synthetic_dataset = np.array(synthetic_dataset)
    synthetic_dataset = [i.tolist() for i in synthetic_dataset]
    synthetic_dataset = np.array(synthetic_dataset)
    
    civa_dataset = np.array(civa_dataset)
    civa_dataset = [i.tolist() for i in civa_dataset]
    civa_dataset = np.array(civa_dataset)

    # plt.figure()
    # plt.imshow(synthetic_dataset[5])
    # plt.title('np')
    # plt.colorbar()
    
    print('shape ', np.shape(synthetic_dataset))    
    print('type ', type(synthetic_dataset))
    save_path = r"C:/Users/CUE-ML/shaun/Datasets/CIVA/" + model_iter + "synthetic_dataset"
    np.save(save_path, synthetic_dataset)
    
    return synthetic_dataset, civa_dataset
        
synthetic_dataset, civa_dataset = generate_synthetic_data(Gen_AB, Gen_BA)   
##############################################
# Execute
##############################################

# validate(
#     Gen_BA=Gen_BA,
#     Gen_AB=Gen_AB,
# )

# plot_losses(model_path)
# plotAllImages(civa_dataset)
plotAllImages(synthetic_dataset, civa_dataset)