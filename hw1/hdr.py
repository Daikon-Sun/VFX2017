import cv2, sys
import numpy as np

prefix = 'pics1/DSC049'

img_fn = [prefix+str(i)+'.JPG' for i in range(55, 66)]
img_list = [cv2.imread(fn) for fn in img_fn]

choice = 2

if choice == 0:
  exposure_times = np.logspace(-13, -3, num=11, base=2, dtype=np.float32)
  merge_robertson = cv2.createMergeRobertson()
  hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())
  tonemap2 = cv2.createTonemapDurand(gamma=1)
  res_robertson = tonemap2.process(hdr_robertson.copy())
  res_robertson_8bit = np.clip(res_robertson*255, 0, 255).astype('uint8')
  cv2.imwrite("testing.jpg", res_robertson_8bit)

elif choice == 1:
  exposure_times = np.logspace(-5, 5, num=11, base=2, dtype=np.float32)
  merge_debvec = cv2.createMergeDebevec()
  hdr_debvec = merge_debvec.process(img_list, times=exposure_times.copy())
  tonemap1 = cv2.createTonemapDurand(gamma=3)
  res_debvec = tonemap1.process(hdr_debvec.copy())
  res_debvec_8bit = np.clip(res_debvec*255, 0, 255).astype('uint8')
  cv2.imwrite("testing2.jpg", res_debvec_8bit)
else:
  merge_mertens = cv2.createMergeMertens()
  res_mertens = merge_mertens.process(img_list)
  res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
  cv2.imwrite("mertens.jpg", res_mertens_8bit)
