#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_eman2/bin/python
################################################################################
# imports
print('Beginning imports...')
import numpy as np
import mrcfile
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
print('Imports finished. Beginning script...')
################################################################################
def pw_spectra(stack):
	pw = []
	for i in tqdm(range(0, len(stack))):
		pw.append(np.fft.fftshift(np.fft.fft2(stack[i])))
	return np.asarray(pw)

################################################################################
file_names = sorted(glob.glob('/rugpfs/fs0/cem/store/achin/fsgs/micrographs_bin4/*.mrc'))
box_size = 256
half_box_size = int(box_size / 2)

# get dimensions of first file
with mrcfile.open(file_names[0]) as mrc:
	temp = mrc.data

xdim, ydim = temp.shape[0], temp.shape[1]

# Load in 1000 random samples
rands = np.random.randint(0,len(file_names)-1,len(file_names))
rands2 = np.random.randint(int(xdim*0.25),int(xdim*0.75),len(file_names))
rands3 = np.random.randint(int(ydim*0.25),int(ydim*0.75),len(file_names))

crop_holder = []
print('Cropping micrograph sections...')
for i in tqdm(range(0, 2000)):
    with mrcfile.open(file_names[rands[i]], 'r') as mrc:
	    crop_holder.append(mrc.data[rands2[i]-half_box_size:rands2[i]+half_box_size, rands3[i]-half_box_size:rands3[i]+half_box_size])


crop_holder = np.asarray(crop_holder)

print('Finished cropping out micrograph boxes. Computing power spectra...')
pw = pw_spectra(crop_holder)

print('Calculating average power spectrum of real data...')
amp_mult = np.average(np.abs(pw),axis=0)
amp_mult[128,128] = 0
amp_mult_std = np.std(np.abs(pw),axis=0)
amp_mult_std[128,128] = 0


fig, ax = plt.subplots(1,3)
ax[0].imshow(amp_mult, cmap=plt.cm.gray)
ax[1].imshow(amp_mult - amp_mult_std, cmap=plt.cm.gray)
ax[2].imshow(amp_mult + amp_mult_std, cmap=plt.cm.gray)
plt.show()


with mrcfile.new('./empirical_pink_noise_model_k255e.mrcs', overwrite=True) as mrc:
	mrc.set_data(np.asarray([amp_mult, amp_mult_std]).astype('float32'))


