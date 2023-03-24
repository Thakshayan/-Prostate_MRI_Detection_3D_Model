import nibabel as nib
import random
import elasticdeform 
import numpy as np
import nibabel as nib

from scipy.ndimage import affine_transform, rotate, zoom
from skimage.exposure import adjust_gamma, rescale_intensity
from skimage.util import random_noise


def random_flip_rotate_zoom_translate(image, label):
    angle = random.uniform(-5, 5)
    flip_axes = [ i for i in range(2) if i!=2 and np.random.choice([0, 1]) == 1]
    translate = (random.uniform(-10,10), random.uniform(-10,10))
    matrix = np.array([[1, 0, 0, translate[0]], [0, 1, 0, translate[1]], [0, 0, 1,  0], [0, 0, 0, 1]])

    imgvol = np.array(image.dataobj)
    lblvol = np.array(label.dataobj)

    # Rotate
    img = rotate(imgvol, angle, reshape=False)
    lbl = rotate(lblvol, angle, reshape=False)

    # Randomly flip the image and label along one or more axes, except for the z-axis
    img = np.flip(img, axis=flip_axes)
    lbl = np.flip(lbl, axis=flip_axes)

    # zoom
    zoom_factor = random.uniform(0.8, 1.2)
    img = zoom(img, zoom_factor, order=1)
    lbl = zoom(lbl, zoom_factor, order=0)

    # Translate
    img = affine_transform(img, matrix, order=1)
    lbl = affine_transform(lbl, matrix, order=0)

    image = nib.Nifti1Image(img, image.affine)
    label = nib.Nifti1Image(lbl, label.affine)
    return image, label

def random_flip_rotate_zoom_shear_translate(image, label):
    
    angle = random.uniform(-5, 5)
    flip_axes = [ i for i in range(2) if i!=2 and np.random.choice([0, 1]) == 1]
    translate = (random.uniform(-10,10), random.uniform(-10,10))
    shear = random.uniform(-0.2,0.2)
    matrix = np.array([[1, shear, 0, translate[0]], [0, 1, 0, translate[1]], [0, 0, 1, 0], [0, 0, 0, 1]])
    

    imgvol = np.array(image.dataobj)
    lblvol = np.array(label.dataobj)

    # Rotate
    img = rotate(imgvol, angle, reshape=False)
    lbl = rotate(lblvol, angle, reshape=False)

    # Randomly flip the image and label along one or more axes, except for the z-axis
    img = np.flip(img, axis=flip_axes)
    lbl = np.flip(lbl, axis=flip_axes)

    # zoom
    zoom_factor = random.uniform(0.8, 1.2)
    img = zoom(img, zoom_factor, order=1)
    lbl = zoom(lbl, zoom_factor, order=0)

    # Translate
    img = affine_transform(img, matrix, order=1)
    lbl = affine_transform(lbl, matrix, order=0)

    image = nib.Nifti1Image(img, image.affine)
    label = nib.Nifti1Image(lbl, label.affine)
    return image, label

def random_flip_rotate_translate_deform(image, label):
    angle = random.uniform(-5, 5)
    flip_axes = [ i for i in range(2) if i!=2 and np.random.choice([0, 1]) == 1]
    sigma = random.choice([2, 3])
    translate = (random.uniform(-10,10), random.uniform(-10,10))
    matrix = np.array([[1, 0, 0, translate[0]], [0, 1, 0, translate[1]], [0, 0, 1,  0], [0, 0, 0, 1]])

    imgvol = np.array(image.dataobj)
    lblvol = np.array(label.dataobj)

    # Rotate
    img = rotate(imgvol, angle, reshape=False)
    lbl = rotate(lblvol, angle, reshape=False)

    # Randomly flip the image and label along one or more axes, except for the z-axis
    img = np.flip(img, axis=flip_axes)
    lbl = np.flip(lbl, axis=flip_axes)

    # zoom
    zoom_factor = random.uniform(0.8, 1.2)
    img = zoom(img, zoom_factor, order=1)
    lbl = zoom(lbl, zoom_factor, order=0)

    # Elastic deformation
    [img, lbl] = elasticdeform.deform_random_grid([img, lbl], sigma=sigma, axis=[(0, 1), (0, 1)], order=[1, 0], mode='constant')
    


    # Translate
    img = affine_transform(img, matrix, order=1)
    lbl = affine_transform(lbl, matrix, order=0)

    image = nib.Nifti1Image(img, image.affine)
    label = nib.Nifti1Image(lbl, label.affine)
    return image, label



def random_flip_rotate_zoom_translate_brightness(image, label):
    angle = random.uniform(-5, 5)
    flip_axes = [i for i in range(2) if i != 2 and np.random.choice([0, 1]) == 1]
    translate = (random.uniform(-10, 10), random.uniform(-10, 10))
    matrix = np.array([[1, 0, 0, translate[0]], [0, 1, 0, translate[1]], [0, 0, 1, 0], [0, 0, 0, 1]])

    imgvol = np.array(image.dataobj)
    lblvol = np.array(label.dataobj)

    # Rotate
    img = rotate(imgvol, angle, reshape=False)
    lbl = rotate(lblvol, angle, reshape=False)

    # Randomly flip the image and label along one or more axes, except for the z-axis
    img = np.flip(img, axis=flip_axes)
    lbl = np.flip(lbl, axis=flip_axes)

    # Zoom
    zoom_factor = random.uniform(0.8, 1.2)
    img = zoom(img, zoom_factor, order=1)
    lbl = zoom(lbl, zoom_factor, order=0)

    # Translate
    img = affine_transform(img, matrix, order=1)
    lbl = affine_transform(lbl, matrix, order=0)

    # Brightness
    gamma = random.uniform(0.8, 1.2)
    min_value = np.min(img)
    img = img - min_value
    img = adjust_gamma(img, gamma)

    # Contrast
    low = np.percentile(img, 10)
    high = np.percentile(img, 98)
    img = rescale_intensity(img, in_range=(low, high))

    image = nib.Nifti1Image(img, image.affine)
    label = nib.Nifti1Image(lbl, label.affine)
    return image, label

def random_flip_rotate_noise(image, label):
    angle = random.uniform(-5, 5)
    imgvol = np.array(image.dataobj)
    lblvol = np.array(label.dataobj)

    # Rotate
    img = rotate(imgvol, angle, reshape=False)
    lbl = rotate(lblvol, angle, reshape=False)

    # Randomly flip the image and label along one or more axes
    flip_axes = [i for i in range(2) if i != 2 and np.random.choice([0, 1]) == 1]
    img = np.flip(img, axis=flip_axes)
    lbl = np.flip(lbl, axis=flip_axes)

    # Zoom
    zoom_factor = random.uniform(0.8, 1.2)
    img = zoom(img, zoom_factor, order=1)
    lbl = zoom(lbl, zoom_factor, order=0)

    # Add noise
    # Gaussian noise
    img = random_noise(img, mode='gaussian', var=0.000001, clip=False)
    # Salt & Pepper noise
    img = random_noise(img, mode='s&p', salt_vs_pepper=0.5, amount=0.0000005, clip=False)

    # Translate
    # matrix = np.array([[1, 0, 0, translate[0]], [0, 1, 0, translate[1]], [0, 0, 1, 0], [0, 0, 0, 1]])
    # img = affine_transform(img, matrix, order=1)
    # lbl = affine_transform(lbl, matrix, order=0)

    image = nib.Nifti1Image(img, image.affine)
    label = nib.Nifti1Image(lbl, label.affine)
    return image, label
