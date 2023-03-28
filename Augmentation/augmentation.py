import nibabel as nib
import random
import elasticdeform 
import numpy as np
import nibabel as nib

from scipy.ndimage import affine_transform, rotate, zoom
from skimage.exposure import adjust_gamma, rescale_intensity
from skimage.util import random_noise

def verticalFlip(image, label):
    imgvol = np.array( image.dataobj )
    lblvol = np.array( label.dataobj )
    img = np.flipud(imgvol)
    lbl = np.flipud(lblvol)
    image = nib.Nifti1Image ( img, image.affine )
    label = nib.Nifti1Image ( lbl, label.affine )
    return image, label

def horizontalFlip(image, label):
    imgvol = np.array( image.dataobj )
    lblvol = np.array( label.dataobj )
    img = np.fliplr(imgvol)
    lbl = np.fliplr(lblvol)
    image = nib.Nifti1Image ( img, image.affine )
    label = nib.Nifti1Image ( lbl, label.affine )
    return image, label


def rotated(image, label ):
    angle =  random.uniform(-15, 15)
    imgvol = np.array( image.dataobj )
    lblvol = np.array( label.dataobj )
    img = rotate(imgvol, angle)
    lbl = rotate(lblvol, angle)
    image = nib.Nifti1Image ( img, image.affine )
    label = nib.Nifti1Image ( lbl, label.affine )
    return image, label


def elasticDeformation(image, label):

    # Set elastic deformation parameters
    sigma = 2  # Elastic deformation intensity 2,3 for prostate 2-5 for mri 8,10 for experiments
    order = 3   # Interpolation order
    mode='constant'  # Boundary condition for interpolation
    
    imgvol = np.array( image.dataobj )
    lblvol = np.array( label.dataobj )
    img = elasticdeform.deform_random_grid(imgvol, sigma=sigma,  order=3, mode=mode)
    lbl = elasticdeform.deform_random_grid(lblvol, sigma=sigma,  order=0, mode=mode)
    image = nib.Nifti1Image ( img, image.affine )
    label = nib.Nifti1Image ( lbl, label.affine )
    return image, label
   

def noise(image, label):
    # modes = ['s&p','gaussian','speckle']
    imgvol = np.array( image.dataobj )
    noisy_image = random_noise(imgvol, mode='gaussian', var=0.000001, clip=False)
    noisy_image = random_noise(noisy_image, mode='s&p', salt_vs_pepper=0.5, amount=0.0000005, clip=False)
    image = nib.Nifti1Image ( noisy_image, image.affine )
    return image, label





def changeContrast(image, label):
    imgvol = np.array(image.dataobj)
    lblvol = np.array(label.dataobj)

    # Brightness
    gamma = random.uniform(0.8, 1.2)
    min_value = np.min(imgvol)
    img = imgvol - min_value
    img = adjust_gamma(img, gamma)

    # Contrast
    low = np.percentile(img, 10)
    high = np.percentile(img, 98)
    img = rescale_intensity(img, in_range=(low, high))

    lbl = lblvol

    image = nib.Nifti1Image(img, image.affine)
    label = nib.Nifti1Image(lbl, label.affine)
    return image, label


def translate(image, label):

    translate = (random.uniform(-10, 10), random.uniform(-10, 10))
    matrix = np.array([[1, 0, 0, translate[0]], [0, 1, 0, translate[1]], [0, 0, 1, 0], [0, 0, 0, 1]])

    imgvol = np.array(image.dataobj)
    lblvol = np.array(label.dataobj)

    # Translate
    img = affine_transform(imgvol, matrix, order=1)
    lbl = affine_transform(lblvol, matrix, order=0)

    image = nib.Nifti1Image(img, image.affine)
    label = nib.Nifti1Image(lbl, label.affine)
    return image, label


def sheartranslate(image, label):

    translate = (random.uniform(-10,10), random.uniform(-10,10))
    shear = random.uniform(-0.2,0.2)
    matrix = np.array([[1, shear, 0, translate[0]], [0, 1, 0, translate[1]], [0, 0, 1, 0], [0, 0, 0, 1]])
    
    imgvol = np.array(image.dataobj)
    lblvol = np.array(label.dataobj)

    # Translate
    img = affine_transform(imgvol, matrix, order=1)
    lbl = affine_transform(lblvol, matrix, order=0)

    image = nib.Nifti1Image(img, image.affine)
    label = nib.Nifti1Image(lbl, label.affine)
    return image, label

def randomFlip(image, label):
    flip_axes = [ i for i in range(2) if i!=2 and np.random.choice([0, 1]) == 1]

    imgvol = np.array(image.dataobj)
    lblvol = np.array(label.dataobj)

    # Randomly flip the image and label along one or more axes, except for the z-axis
    img = np.flip(imgvol, axis=flip_axes)
    lbl = np.flip(lblvol, axis=flip_axes)

    image = nib.Nifti1Image(img, image.affine)
    label = nib.Nifti1Image(lbl, label.affine)
    return image, label