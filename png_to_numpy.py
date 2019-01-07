from PIL import Image
import numpy as np
import glob

#for i, im_path in enumerate(glob.glob("./A/train/*.png")):
#    im_frame = Image.open(im_path)
#    np_frame = np.array(im_frame.getdata()).reshape(im_frame.size)
#    np.save("./A/np/train/{:03d}.npy".format(i), np_frame)
#    print(np_frame.shape)
arrays = []
for i, im_path in enumerate(glob.glob("./A/np/train/*.npy")):
    arr = np.load(im_path)
    arrays.append(arr)
    print(arr.shape)
    if arr.shape != (512, 512):
        print("Not the same size! Expected (512, 512).")
all_arrays = np.zeros((len(arrays), 1, 512, 512), dtype="float32")
for i, arr in enumerate(arrays):
    all_arrays[i, ...] = arr
np.save("./A/np/train/all.npy", all_arrays)
print(all_arrays.shape)
