import numpy as np
import argparse
import matplotlib.pyplot as plt

from glob import glob
from scipy.misc import imread
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from skimage.color import rgb2gray


def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--pred_data', default='/home/song/view_synthesis/synpts/results/fine')
    parser.add_argument('--real_data', default='/home/song/view_synthesis/synpts/results/real')
    parser.add_argument('--debug', default=0, help='Debug', type=int)
    args = parser.parse_args()
    return args

def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.mean(np.abs(img_true - img_test))


args = parse_args()
for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))


psnr = []
ssim = []
mae = []
names = []
index = 1

pred_files = sorted(glob(args.pred_data + '/*.png'))
real_files = sorted(glob(args.real_data + '/*.png'))
assert len(pred_files) == len(real_files)
for i in range(len(real_files)):

    img_gt = (imread(real_files[i]) / 255.0).astype(np.float32)
    img_pred = (imread(pred_files[i]) / 255.0).astype(np.float32)

    img_gt = rgb2gray(img_gt)
    img_pred = rgb2gray(img_pred)

    if args.debug != 0:
        plt.subplot('121')
        plt.imshow(img_gt)
        plt.title('Groud truth')
        plt.subplot('122')
        plt.imshow(img_pred)
        plt.title('Output')
        plt.show()

    psnr.append(compare_psnr(img_gt, img_pred, data_range=1))
    ssim.append(compare_ssim(img_gt, img_pred, data_range=1, win_size=51))
    mae.append(compare_mae(img_gt, img_pred))
    if np.mod(index, 1000) == 0:
        print(
            str(index) + ' images processed',
            "PSNR: %.4f" % round(np.mean(psnr), 4),
            "SSIM: %.4f" % round(np.mean(ssim), 4),
            "MAE: %.4f" % round(np.mean(mae), 4),
        )
    index += 1

# np.savez('metrics.npz', psnr=psnr, ssim=ssim, mae=mae, names=names)
print(
    "PSNR: %.4f\n" % round(np.mean(psnr), 4),
    "PSNR Variance: %.4f\n" % round(np.var(psnr), 4),
    "SSIM: %.4f\n" % round(np.mean(ssim), 4),
    "SSIM Variance: %.4f\n" % round(np.var(ssim), 4),
    "MAE: %.4f\n" % round(np.mean(mae), 4),
    "MAE Variance: %.4f\n" % round(np.var(mae), 4)
)
