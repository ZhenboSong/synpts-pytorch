import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
from skimage.feature import canny, ORB
import glob


class DeMonLoader(Dataset):
    def __init__(self, root_dir, data_dir, im_size, pc_size=None):
        self.im_size = im_size
        if pc_size is None:
            self.pc_size = int(im_size[0]*im_size[1]/16)
        else:
            self.pc_size = pc_size
        self.file_dicts = []
        f = open(data_dir, "r")
        for line in f:
            line = line.strip("\n")
            self.file_dicts.append(root_dir + '/' + line)
        f.close()

    def generate_point_cloud(self, depth, K):
        h, w = depth.shape
        u = np.tile(np.arange(0, w, dtype=np.float64), h).reshape(h, w)
        v = np.tile(np.arange(0, h, dtype=np.float64).reshape(h, 1), w)
        fx, fy, cx, cy = K
        x = np.expand_dims(depth * (u - cx) / fx, axis=2)
        y = np.expand_dims(depth * (v - cy) / fy, axis=2)
        z = np.expand_dims(depth, axis=2)
        xyz = np.concatenate((x, y, z), axis=2)
        return xyz

    def sample_pc_curb(self, rgbmap, xyzmap, size):
        ds_rgb = cv2.resize(rgbmap, (self.im_size[1], self.im_size[0]), interpolation=cv2.INTER_NEAREST)
        ds_xyz = cv2.resize(xyzmap, (self.im_size[1], self.im_size[0]), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(ds_rgb, cv2.COLOR_RGB2GRAY)
        curb_mask = canny(gray, sigma=np.random.randint(1, 4))
        mask = curb_mask & (ds_xyz[:, :, 2] > 0.1)
        sampled_pc = ds_xyz[mask]
        sampled_rgb = ds_rgb[mask] / 255.
        n, _ = sampled_pc.shape
        sampled_cpc = np.concatenate((sampled_pc, sampled_rgb), axis=1)
        if n < size:
            sub_mask = (~curb_mask) & (ds_xyz[:, :, 2] > 0.0)
            add_cpc = self.sample_pc_random(ds_rgb, ds_xyz, size - n, sub_mask)
            sampled_cpc = np.concatenate((sampled_cpc, add_cpc), axis=0)
        return sampled_cpc

    def sample_pc_random(self, rgbmap, xyzmap, size, mask=None):
        ds_rgb = cv2.resize(rgbmap, (self.im_size[1], self.im_size[0]), interpolation=cv2.INTER_NEAREST)
        ds_xyz = cv2.resize(xyzmap, (self.im_size[1], self.im_size[0]), interpolation=cv2.INTER_NEAREST)
        if mask is None:
            mask = ds_xyz[:, :, 2] > 0.1
        flat_rgb = ds_rgb[mask].reshape(-1, 3)
        flat_xyz = ds_xyz[mask].reshape(-1, 3)
        n = flat_rgb.shape[0]
        sample_loc = np.random.randint(0, n, size)
        sampled_rgb = flat_rgb[sample_loc, :] / 255.
        sampled_pc = flat_xyz[sample_loc, :]
        sampled_cpc = np.concatenate((sampled_pc, sampled_rgb), axis=1)
        return sampled_cpc

    def sample_pc_uniform(self, rgbmap, xyzmap, size):
        h, w, _ = rgbmap.shape
        ds_rgb = cv2.resize(rgbmap, (int(w/4), int(h/4)), interpolation=cv2.INTER_NEAREST)
        ds_xyz = cv2.resize(xyzmap, (int(w/4), int(h/4)), interpolation=cv2.INTER_NEAREST)
        mask = ds_xyz[:, :, 2] > 0.1
        sampled_rgb = ds_rgb[mask] / 255.
        sampled_pc = ds_xyz[mask]
        sampled_cpc = np.concatenate((sampled_pc, sampled_rgb), axis=1)
        n, _ = sampled_cpc.shape
        if n < size:
            chosen_mask = np.tile(np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.bool),
                               (int(h/4), int(w/4)))
            sub_mask = (~chosen_mask) & (xyzmap[:, :, 2] > 0.0)
            add_cpc = self.sample_pc_curb(rgbmap, xyzmap, size - n, sub_mask)
            sampled_cpc = np.concatenate((sampled_cpc, add_cpc), axis=0)
        return sampled_cpc

    def mimic_sfm_sample(self, rgbmap, xyzmap, size):
        ds_rgb = cv2.resize(rgbmap, (self.im_size[1], self.im_size[0]), interpolation=cv2.INTER_NEAREST)
        ds_xyz = cv2.resize(xyzmap, (self.im_size[1], self.im_size[0]), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(ds_rgb, cv2.COLOR_RGB2GRAY)

        mask_range = ds_xyz[:, :, 2] > 0.1  # in range is True

        detector = ORB(n_keypoints=size*2)
        detector.detect(gray)
        kpt = detector.keypoints
        mask_kpt = np.zeros_like(gray).astype(np.bool)
        mask_kpt[np.round(kpt[:, 0]).astype(np.int), np.round(kpt[:, 1]).astype(np.int)] = True  # kpt is True
        kpt_pt3 = ds_xyz[mask_kpt]
        kpt_color = ds_rgb[mask_kpt]/255.
        kpt_cpc = np.concatenate((kpt_pt3, kpt_color), axis=1)

        mask_curb = canny(gray, sigma=np.random.randint(1, 4))  # curb is True
        mask_curb_not_kpt = mask_curb & mask_range & (~mask_kpt)  # curb is True, kpt is False
        curb_pt3 = ds_xyz[mask_curb_not_kpt]
        curb_color = ds_rgb[mask_curb_not_kpt] / 255.
        curb_cpc = np.concatenate((curb_pt3, curb_color), axis=1)

        if curb_cpc.shape[0] + kpt_cpc.shape[0] < size:
            # at least randomly sample one
            mask_not_curb_not_kpt = (~mask_curb) & mask_range & (~mask_kpt)  # curb is False, kpt is False
            random_cpc = self.sample_pc_random(rgbmap, xyzmap,
                                               size-curb_cpc.shape[0]-kpt_cpc.shape[0], mask_not_curb_not_kpt)
            sampled_cpc = np.concatenate((kpt_cpc, curb_cpc, random_cpc), axis=0)

        elif kpt_cpc.shape[0] >= size:
            sample_loc = np.random.randint(0, kpt_cpc.shape[0], size)
            sampled_cpc = kpt_cpc[sample_loc, :]

        else:
            sample_loc = np.random.randint(0, curb_pt3.shape[0], size-kpt_pt3.shape[0])
            sampled_cpc = np.concatenate((kpt_cpc, curb_cpc[sample_loc, :]), axis=0)

        return sampled_cpc

    def read_image(self, image_dir):
        image = cv2.imread(image_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def read_depth(self, depth_dir):
        suffix = depth_dir.split('.')[-1]
        if suffix == 'npy':
            depth = np.load(depth_dir)
        else:
            depth = cv2.imread(depth_dir)
        depth = np.clip(depth, a_min=1e-7, a_max=5000.0)
        return depth

    def pick_top2_noblur(self, file_dir):
        image_files = sorted(glob.glob(file_dir + '/*.jpg'), reverse=False)
        scores = list()
        for i in range(len(image_files)):
            gray = cv2.imread(image_files[i], 0)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores.append(score)
        sorted_idx = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        return sorted_idx[0], sorted_idx[1]

    def uniform_pc_size(self, pc, size):
        n, _ = pc.shape
        t_pc = pc.transpose(0, 1).unsqueeze(0)
        if n < size:
            t_pc = F.interpolate(t_pc, size, mode='linear', align_corners=True)
        else:
            t_pc = F.interpolate(t_pc, size, mode='nearest')
        return t_pc.squeeze()

    def get_sample(self, file_dir):
        # load data from files
        idx0, idx1 = self.pick_top2_noblur(file_dir)
        image0 = self.read_image('%s/%04d.jpg' % (file_dir, idx0))
        image1 = self.read_image('%s/%04d.jpg' % (file_dir, idx1))
        depth0 = self.read_depth('%s/%04d.npy' % (file_dir, idx0))
        depth1 = self.read_depth('%s/%04d.npy' % (file_dir, idx1))
        calib = np.loadtxt(file_dir + '/cam.txt').reshape(3, 3)
        fx = calib[0, 0]
        cx = calib[0, 2]
        fy = calib[1, 1]
        cy = calib[1, 2]
        poses = np.loadtxt(file_dir + '/poses.txt').reshape(-1, 3, 4)
        pose_0 = np.vstack((poses[idx0], [0, 0, 0, 1.]))
        pose_1 = np.vstack((poses[idx1], [0, 0, 0, 1.]))
        warp_0_1 = np.dot(pose_1, np.linalg.inv(pose_0))  # camera to world, pose from 1 to 0, warping from 0 to 1

        # sample point cloud on original size
        xyz0 = self.generate_point_cloud(depth0, [fx, fy, cx, cy])
        sampled_cpc = self.mimic_sfm_sample(image0, xyz0, self.pc_size)

        # rescale
        max_range = np.max(np.abs(sampled_cpc[:, :3]))
        sampled_cpc[:, :3] = 5. * sampled_cpc[:, :3] / max_range
        warp_0_1[:, :3] = 5 * warp_0_1[:, :3] / max_range

        # note: resize pytorch (y, x), cv2 (x, y), PIL (x, y)
        if self.im_size[1] == self.im_size[0]:
            size_o = image0.shape
            dw = int((size_o[1] - size_o[0])/2)
            depth0 = depth0[:, dw:int(dw+size_o[0])]
            depth1 = depth1[:, dw:int(dw + size_o[0])]
            image0 = image0[:, dw:int(dw + size_o[0]), :]
            image1 = image1[:, dw:int(dw + size_o[0]), :]
            cx -= dw
        size_o = image0.shape
        scalor_x = self.im_size[1] / size_o[1]
        scalor_y = self.im_size[0] / size_o[0]
        depth0 = cv2.resize(depth0, (self.im_size[1], self.im_size[0]), interpolation=cv2.INTER_LINEAR)
        depth1 = cv2.resize(depth1, (self.im_size[1], self.im_size[0]), interpolation=cv2.INTER_LINEAR)
        image0 = cv2.resize(image0, (self.im_size[1], self.im_size[0]), interpolation=cv2.INTER_LINEAR)
        image1 = cv2.resize(image1, (self.im_size[1], self.im_size[0]), interpolation=cv2.INTER_LINEAR)
        fx *= scalor_x
        cx *= scalor_x
        fy *= scalor_y
        cy *= scalor_y

        # to tensor
        depth0 = torch.from_numpy(depth0.astype(np.float32))
        depth1 = torch.from_numpy(depth1.astype(np.float32))
        image0 = torch.from_numpy(image0.transpose(2, 0, 1).astype(np.float32)) / 255.
        image1 = torch.from_numpy(image1.transpose(2, 0, 1).astype(np.float32)) / 255.
        sampled_cpc = torch.from_numpy(sampled_cpc.astype(np.float32))
        warp_0_1 = torch.from_numpy(warp_0_1.astype(np.float32))
        calib = torch.from_numpy(np.array([fx, fy, cx, cy], dtype=np.float32))

        # uniform the size of point cloud
        uniform_cpc = self.uniform_pc_size(sampled_cpc, self.pc_size)

        sample = {'image': [image0, image1],
                  'depth': [depth0, depth1],
                  'points': uniform_cpc,
                  'camera': calib,
                  'warp': warp_0_1}
        return sample

    def __len__(self):
        return len(self.file_dicts)

    def __getitem__(self, idx):
        sample = self.get_sample(self.file_dicts[idx])
        return sample


if __name__ == '__main__':
    root_dir = '/home/song/Documents/data/demon'
    data_dir = 'filenames/demon_debug.txt'

    data_set = DeMonLoader(root_dir, data_dir, im_size=(256, 256), pc_size=1024*4)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=2, shuffle=False, num_workers=8)
    # from torchvision.transforms import transforms
    # from PIL import ImageDraw
    # import models.geometry as geo
    count = 0
    for data in data_loader:
        # org_image0 = data['image'][0]
        # pointcloud = data['points']
        # K = data['camera']
        #
        # B, C, H, W = org_image0.shape
        # pt30 = pointcloud[:, :3, :]
        # f0 = pointcloud[:, 3:, :]
        # pt20 = geo.project3Dto2D(pt30, K)  # B, 2, N
        # sort_data0, sort_id0 = torch.sort(pt30[:, 2, :], dim=-1, descending=True)
        # new_image1 = geo.generateImage(pt20, f0, sort_id0, [H, W])
        #
        # image_a = transforms.ToPILImage()(org_image0[0].data.cpu()).convert('RGB')
        # draw = ImageDraw.Draw(image_a)
        # for i in range(pt20.shape[2]):
        #     draw.point((pt20[0, 0, i], pt20[0, 1, i]), fill='red')
        # image_a.save("image00.png")
        # break


        print(count, data['points'].shape)
        count += 1
        # image_a = transforms.ToPILImage()(data['image'][0][0]).convert('RGB')
        # image_b = data['depth'][1][0].numpy()
        # cv2.imwrite('depth.png', image_b*51)
        # image_a.save("image.png")
