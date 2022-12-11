import photomosaic as pm
from skimage.io import imread, imsave


image = imread('/Users/coryrandolph/Documents/Programing/mosaic_generator/src/input_images/Pikachu.jpeg')


# Generate a collection of solid-color square images.
# pm.rainbow_of_squares('pool/')

# Analyze the collection (the "pool") of images.
pool = pm.make_pool('/Users/coryrandolph/Documents/Programing/mosaic_generator/src/mosaic_generator/pool/*.png')

# Create a mosiac with 30x30 tiles.
mos = pm.basic_mosaic(image, pool, (30, 30))

# Save the mosaic
imsave('/Users/coryrandolph/Documents/Programing/mosaic_generator/src/output_mosaics/mosaic.png', mos)

# import matplotlib.pyplot as plt
# # plt.imshow(mos)

# from skimage import data
# # image = data.chelsea()  # cat picture!
# plt.imshow(image)
# plt.savefig('../output_mosaics/mosaic.png')

# import os
# print(os.getcwd()) 

