import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numpy as np
import torch
import nibabel as nib
import matplotlib.animation as animation
from celluloid import Camera
from IPython.display import HTML
from model.baseline_lightning import LightningSegmentation


def plot_image_by_slice(img_data, number_fig=5):
    fig_rows = number_fig
    fig_cols = number_fig
    n_subplots = fig_rows * fig_cols
    n_slice = img_data.shape[0]
    step_size = n_slice // n_subplots
    plot_range = n_subplots * step_size
    start_stop = int((n_slice - plot_range) / 2)

    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=[20, 20])

    for idx, img in enumerate(range(start_stop, plot_range, step_size)):
        axs.flat[idx].imshow(ndi.rotate(img_data[img, :, :], 90), cmap="gray")
        axs.flat[idx].axis("off")

    plt.tight_layout()
    plt.show()


def plot_from_loader(cts, masks):
    ct, mask = cts[2], masks[2]
    plt.imshow(ct.squeeze(), cmap="bone")
    mask_ = np.ma.masked_where(mask == 0, mask)
    plt.imshow(mask_.squeeze(), alpha=0.5, cmap="autumn")
    plt.show()


def plot_mask_by_i(ct, mask, i):
    plt.imshow(ct[:, :, i], cmap="bone")
    mask_ = np.ma.masked_where(mask[:, :, i] == 0, mask[:, :, i])
    plt.imshow(mask_, alpha=0.5, cmap="autumn")
    plt.show()


def plot_load_numpy(file_data, file_mask):
    slice = np.load(file_data)
    mask = np.load(file_mask)
    fig, axis = plt.subplots(1, 2, figsize=(8, 8))
    axis[0].imshow(slice, cmap="bone")
    mask_ = np.ma.masked_where(mask == 0, mask)
    axis[1].imshow(slice, cmap="bone")
    axis[1].imshow(mask_, cmap="autumn")
    plt.show()


def make_mri_gif(model_type, val_path, mri_path):
    # Load the model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = LightningSegmentation.load_from_checkpoint(val_path, model=model_type)
    
    # Load the MRI file
    mri = nib.load(mri_path)
    mri_data = mri.get_fdata()
    
    # Make sure the MRI data is in the right shape for the model (CHW)
    mri_data = np.transpose(mri_data, (2, 0, 1)) # This may change depending on your MRI and model
    mri_data = torch.from_numpy(mri_data).float()
    mri_data = mri_data.to(device)
    
    # Run the model
    with torch.no_grad():
        mask = model(mri_data).squeeze(0)
        
    # Transpose back to HWC
    mask = mask.cpu().numpy().transpose(1, 2, 0)
    mask = (mask > 0.5)  # Apply a threshold to get a binary mask
    
    # Initialize the camera
    fig = plt.figure()
    camera = Camera(fig)

    # Loop through each slice
    for i in range(0, mri_data.shape[2], 2):  # axial view
        plt.imshow(mri_data[:,:,i], cmap="bone")
        mask_ = np.ma.masked_where(mask[:,:,i]==0, mask[:,:,i])
        plt.imshow(mask_, alpha=0.5, cmap="autumn")
        plt.axis("off")
        camera.snap()

    # Create the animation
    animation = camera.animate()
    
    return HTML(animation.to_html5_video())

def box_plot(unet_dice, dilated_dice, resnet_dice, attention_dice):

    columns = [unet_dice, dilated_dice, resnet_dice, attention_dice]

    fig, ax = plt.subplots()
    ax.boxplot(columns)
    ax.set_xlabel('Model')
    ax.set_ylabel('Dice score')
    plt.xticks([1, 2, 3, 4], ["U-Net","Dilated U-Net", "ResNet U-Net", "Attention U-Net"])
    plt.show()
