import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2


class ICLDSOLoader(Dataset):
    def __init__(self, root_dir, data_dir, im_size, pc_size=None):
        self.im_size = im_size
        self.root_dir = root_dir
        if pc_size is None:
            self.pc_size = int(im_size[0]*im_size[1]/16)
        else:
            self.pc_size = pc_size
        self.file_dicts = []
        f = open(data_dir, "r")
        for line in f:
            line = line.strip("\n")
            self.file_dicts.append(line)
        f.close()

    def load_one_seq(self, file_dir):
        f = open(file_dir, 'r')
        data = f.read()
        lines = data.replace(",", " ").replace("\t", " ").split("\n")
        sub_dir = lines[0]
        rgb_dir = list()
        depth_dir = list()
        pose = list()
        for i in range(1, len(lines)):
            if not len(lines[i]) > 0:
                continue
            fields = lines[i].split(" ")
            rgb_dir.append(self.root_dir + '/' + sub_dir + '/' + fields[0])
            depth_dir.append(self.root_dir + '/' + sub_dir + '/' + fields[1])
            pose.append([[float(fields[2]), float(fields[3]), float(fields[4]), float(fields[5])],
                         [float(fields[6]), float(fields[7]), float(fields[8]), float(fields[9])],
                         [float(fields[10]), float(fields[11]), float(fields[12]), float(fields[13])]])
        return rgb_dir, depth_dir, pose

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

    def read_image(self, image_dir):
        image = cv2.imread(image_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def randomly_pick2(self, data_num):
        idx = np.random.randint(0, data_num, 2)
        return idx[0], idx[1]

    def uniform_pc_size(self, pc, size):
        n, _ = pc.shape
        t_pc = pc.transpose(0, 1).unsqueeze(0)
        if n < size:
            t_pc = F.interpolate(t_pc, size, mode='linear', align_corners=True)
        else:
            t_pc = F.interpolate(t_pc, size, mode='nearest')
        return t_pc.squeeze()

    def get_sample(self, file_dir):
        rgb, depth, pose = self.load_one_seq(file_dir)

        # load data from files
        idx0, idx1 = self.randomly_pick2(len(rgb))

        image0 = self.read_image(rgb[idx0])
        image1 = self.read_image(rgb[idx1])
        pointcloud = np.load(depth[idx0])

        cx = 320.
        cy = 240.
        fx = 481.20
        fy = 480.0

        pose_0 = np.vstack((pose[idx0], [0, 0, 0, 1.]))
        pose_1 = np.vstack((pose[idx1], [0, 0, 0, 1.]))
        warp_0_1 = np.dot(np.linalg.inv(pose_1), pose_0)  # world to camera transformation

        # sample point cloud
        sample_loc = np.random.randint(0, pointcloud.shape[0], self.pc_size)
        sampled_cpc = pointcloud[sample_loc, :]
        sampled_cpc[:, 3:] = sampled_cpc[:, 3:] / 255.

        # rescale
        max_range = np.max(np.abs(sampled_cpc[:, :3]))
        sampled_cpc[:, :3] = 5. * sampled_cpc[:, :3] / max_range
        warp_0_1[:3, 3] = 5. * warp_0_1[:3, 3] / max_range

        # note: resize pytorch (y, x), cv2 (x, y), PIL (x, y)
        if self.im_size[1] == self.im_size[0]:
            size_o = image0.shape
            dw = int((size_o[1] - size_o[0])/2)
            image0 = image0[:, dw:int(dw + size_o[0]), :]
            image1 = image1[:, dw:int(dw + size_o[0]), :]
            cx -= dw
        size_o = image0.shape
        scalor_x = self.im_size[1] / size_o[1]
        scalor_y = self.im_size[0] / size_o[0]
        image0 = cv2.resize(image0, (self.im_size[1], self.im_size[0]), interpolation=cv2.INTER_LINEAR)
        image1 = cv2.resize(image1, (self.im_size[1], self.im_size[0]), interpolation=cv2.INTER_LINEAR)
        fx *= scalor_x
        cx *= scalor_x
        fy *= scalor_y
        cy *= scalor_y

        # to tensor
        image0 = torch.from_numpy(image0.transpose(2, 0, 1).astype(np.float32)) / 255.
        image1 = torch.from_numpy(image1.transpose(2, 0, 1).astype(np.float32)) / 255.
        sampled_cpc = torch.from_numpy(sampled_cpc.astype(np.float32))  # N, 6
        warp_0_1 = torch.from_numpy(warp_0_1.astype(np.float32))
        calib = torch.from_numpy(np.array([fx, fy, cx, cy], dtype=np.float32))

        # uniform the size of point cloud
        uniform_cpc = self.uniform_pc_size(sampled_cpc, self.pc_size)

        sample = {'image': [image0, image1],
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
    import os
    files = sorted(os.listdir('/home/song/Documents/data/icl_rgbd/dso_data_test'))
    f = open('/home/song/view_synthesis/synpts/data/filenames/icl_test.txt', 'w')
    for x in files:
        f.write('/home/song/Documents/data/icl_rgbd/dso_data_test/%s\n' % x)

    root_dir = '/data/songzb/tumRGBD'
    data_dir = 'filenames/tum_valid.txt'

    data_set = ICLDSOLoader(root_dir, data_dir, im_size=(256, 256))
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=2, shuffle=False, num_workers=4)
    from torchvision.transforms import transforms
    for data in data_loader:
        points = data['points'][0].numpy()  # 6, N
        image_a = transforms.ToPILImage()(data['image'][0][0]).convert('RGB')
        image_b = data['depth'][1][0].numpy()
        image_a.save("image.png")
