import os
from PIL import Image

images_dir = './inputImages'
output_dir = './Spheres/images'
images = sorted(os.listdir(images_dir))

for i in range(len(images)):
    img_i = Image.open(os.path.join(images_dir, images[i]))
    img_i.save(os.path.join(output_dir,f'Picture_{i}.png'))