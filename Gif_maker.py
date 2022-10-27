from PIL import Image, ImageDraw
import os

def flatten(t):
    return [item for sublist in t for item in sublist]

load_pattern = flatten([[
    f'Classification_images/tandir{"/*" * i}.jpg',
    f'Classification_images/tandir{"/*" * i}.tif',
] for i in range(1,2)])

filenames = next(os.walk('Classification_images/kitti_tandir/'), (None, None, []))[2]  # [] if no file

images = []

for i in range(0,40):
    im = Image.open('Classification_images/kitti_tandir/tandir_kitti_' + str(i) + '.jpg')
    images.append(im)
    #if i ==18: break

images[0].save('tandir_0020.gif', append_images=images[1:], optimize=False, duration=200, loop=0, save_all=True)

for i in range(0,40):
    im = Image.open('Classification_images/kitti_raytrace/tandir_kitti_' + str(i) + '.jpg')
    images.append(im)
    #if i ==18: break

images[0].save('raytrace_0020.gif', append_images=images[1:], optimize=False, duration=200, loop=0, save_all=True)