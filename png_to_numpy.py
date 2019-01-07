from PIL import Image
import numpy as np
import glob

for i, im_path in enumerate(glob.glob("./A/train/*.png")):
    im_frame = Image.open(im_path)
    np_frame = np.array(im_frame.getdata()).reshape(im_frame.size)
    np.save("./A/np/train/{:03d}.npy".format(i), np_frame)
    print(np_frame.shape)
