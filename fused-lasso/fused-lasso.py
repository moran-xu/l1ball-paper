 
import torch
import hamiltorch  
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt 

pil_img = Image.open('./abd.png').convert("L").resize((100,80))
 
plt.imshow(pil_img)
img = transforms.ToTensor()(pil_img).unsqueeze_(0)[0] 
img_rgb = torch.zeros(img.shape)
for rgb in range(1):
    img_rgb[rgb] = img[rgb]  + torch.normal(torch.zeros(img_rgb[0].shape), .05).abs()

im1 = transforms.ToPILImage()(img_rgb).convert("L")
def mse_loss(y, w):
    return torch.sum((y - w) ** 2)

def mse_grad(y, w):
    return w - y

def l1_subgrad(weights, indices):
    sum = torch.zeros(len(weights))   + .1 * torch.sign(weights)
    for col in range(indices.shape[1]):
        sum += torch.sign(weights - weights[indices[:,col]])
    return sum

def build_adjacency_matrix(shape):
    """
    Adjacency matrix contains indices of neighbour pixels
    """
    x, y = shape
    a = torch.zeros((x * y, 4), dtype=int)
    for i in range(x):
        for j in range(y):
            ind = i * y + j
            a[ind, 0] = ind if (i - 1) < 0 else y * (i - 1) + j
            a[ind, 1] = ind if (i + 1) >= x else y * (i + 1) + j
            a[ind, 2] = ind if (j - 1) < 0 else y * i + (j - 1)
            a[ind, 3] = ind if (j + 1) >= y else y * i + (j + 1)
    return a

def loss_grad(y, w, adj, l):
    return mse_grad(y, w) +l* l1_subgrad(w, adj)

def denoise(img, n_iter=100, verbose=False, l=.1,lr = .005):
    img_shape = img.shape
    img = img.flatten()
    weights = torch.normal(torch.zeros(img.shape)) * 1E-2
    adjacency_matrix = build_adjacency_matrix(img_shape)
    for it in range(n_iter):
        weights = weights - lr * loss_grad(img, weights, adjacency_matrix, l)
        if verbose and it > 0 and it % 100 == 0:
            print(f"{it}: {mse_loss(img, weights)}")
    return weights.reshape(img_shape)

def proj(img_rgb):
    #display(im)
    im_denoise = torch.zeros(img_rgb.shape)
    for rgb in range(1):
        im = img_rgb[rgb]

        im_denoise[rgb] = denoise(im)
    return(im_denoise)
    
im_denoise = proj(img_rgb)
im2 = transforms.ToPILImage()(im_denoise).convert("L")

from matplotlib.colors import DivergingNorm
norm = DivergingNorm(vmin=0, vcenter=img[0].median(), vmax=img[0].max())
name = ['fused_a','fused_b','fused_c','fused_d']
col='PiYG'
# =============================================================================
# three_img = [img[0], img_rgb[0], im_denoise[0] ,v_plot[0]]
# for i in range(3,4):
#     plt.matshow(three_img[i],norm = norm ,cmap='afmhot')
#     plt.colorbar()
#     plt.axis('off')
#     plt.savefig('C:/Users/laosu/OneDrive - University of Florida/Documents/l1ball_jrssb/fused-lasso/'+name[i])
#     
# 
# =============================================================================
def llh(params):
    theta = proj(params.reshape(img_rgb.shape))
    return( -(params ** 2).sum() / 100. - ((img_rgb - theta) ** 2 ).sum())
           
sampler = hamiltorch.Sampler.HMC_NUTS
num_samples = 500
step_size = .001
params = im_denoise.flatten()#+torch.normal(torch.zeros(len(img_rgb.flatten()))) * .01#
params.requires_grad = True
num_steps_per_sample = 1
hamiltorch.set_random_seed(1)
#M = torch.ones(len(params))
params_hmc = hamiltorch.sample(log_prob_func=llh, params_init=params,  num_samples=num_samples, step_size=step_size, num_steps_per_sample=num_steps_per_sample, desired_accept_rate = .6)
params_hmc = torch.vstack(params_hmc)
hmc_list = []
for _ in  params_hmc[0:,:]:
    hmc_list.append(proj(_.reshape(img_rgb.shape)))

hmc_list  = torch.stack(hmc_list )
variance = hmc_list .var(0)

v_plot = torch.sqrt(variance.reshape(img_rgb.shape)) 
plt.matshow(v_plot[0]) 

mean = hmc_list.mean(0)
mean_plot = mean[0]
m_im = transforms.ToPILImage()(mean_plot).convert("RGB") 
 
 
four_img = [img[0], img_rgb[0], im_denoise[0], v_plot[0]]
title = ['original', 'noise', 'posterior mode', 'pixel-wise variance']
#fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12,4))

for i in range(4): 
    plt.matshow(four_img[i],cmap = 'Oranges')
    plt.axis('off')
    plt.colorbar()
    plt.title(title[i])