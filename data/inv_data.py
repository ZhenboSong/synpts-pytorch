import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import data.colmap.load_data as ld


class InvLoader(Dataset):
    def __init__(self, root_dir, data_dir, im_size, pc_size=None):
        self.im_size = im_size
        self.root_dir = root_dir
        if pc_size is None:
            self.pc_size = int(im_size[0]*im_size[1]/16)
        else:
            self.pc_size = pc_size
        with open(data_dir, 'r') as f:
            self.file_list = [line.strip().split(' ') for line in f]
        for i in range(len(self.file_list)):
            for j in range(6):
                self.file_list[i][j] = root_dir+'/'+self.file_list[i][j]

    def uniform_pc_size(self, pc, size):
        n, _ = pc.shape
        t_pc = pc.transpose(0, 1).unsqueeze(0)
        if n < size:
            t_pc = F.interpolate(t_pc, size, mode='linear', align_corners=True)
        else:
            t_pc = F.interpolate(t_pc, size, mode='nearest')
        return t_pc.squeeze()

    def get_sample(self, idx):
        # load data from files

        pt3 = ld.load_points_xyz(self.file_list[idx][0])
        color = ld.load_points_rgb(self.file_list[idx][1])
        K, R, T, h, w = ld.load_camera(self.file_list[idx][3])

        ptc, calib = ld.pick_points(pt3, color, np.hstack((R, T)), K, [h, w], self.im_size[0], self.im_size[0])

        rgb = ld.scale_crop(ld.load_image(self.file_list[idx][4]), self.im_size[0], self.im_size[0])

        # rescale the point cloud
        ptc[:, :3] = (R.dot(ptc[:, :3].transpose())+T).transpose()
        max_range = np.max(np.abs(ptc[:, :3]))
        ptc[:, :3] = 5. * ptc[:, :3] / max_range

        # randomly sampling
        if ptc.shape[0] > self.pc_size:
            sample_loc = np.random.randint(0, ptc.shape[0], self.pc_size)
            ptc = ptc[sample_loc, :]

        # to tensor
        rgb = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32)) / 255.
        new_ptc = torch.from_numpy(ptc.astype(np.float32))
        calib = torch.from_numpy(calib.astype(np.float32))

        # sample point cloud
        uniform_cpc = self.uniform_pc_size(new_ptc, self.pc_size)

        sample = {'image': [rgb],
                  'points': uniform_cpc,
                  'camera': calib}
        return sample

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.get_sample(idx)
        return sample


def select_data(root_dir, data_dir):
    with open(data_dir, 'r') as f:
        file_list = [line.strip().split(' ') for line in f]
    with open(data_dir, 'r') as f:
        raw_files = [line.strip().split(' ') for line in f]
    for i in range(len(file_list)):
        for j in range(6):
            file_list[i][j] = root_dir + '/' + file_list[i][j]
    f = open('inv_test.txt', 'w')
    count = 0
    mincount = 100000
    for idx in range(len(file_list)):
        pt3 = ld.load_points_xyz(file_list[idx][0])
        color = ld.load_points_rgb(file_list[idx][1])
        K, R, T, h, w = ld.load_camera(file_list[idx][3])
        ptc, calib = ld.pick_points(pt3, color, np.hstack((R, T)), K, [h, w], 256, 256)
        if ptc.shape[0] < mincount:
            mincount = ptc.shape[0]
        if ptc.shape[0] > 4096*2:
            print(count)
            count += 1
            for j in range(6):
                f.write("%s " % raw_files[idx][j])
            f.write('\n')
    f.close()
    print('min size: ', mincount)


if __name__ == '__main__':
    root_dir = '/home/song/Documents/data/invsfm_data'
    data_dir = '/home/song/Documents/data/invsfm_data/anns/demo_5k/inv_test.txt'
    # select_data(root_dir, data_dir)
    data_set = InvLoader(root_dir, data_dir, im_size=(256, 256), pc_size=4096)
    n_img = len(data_set)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False, num_workers=1)

    import models.geometry as geo
    from torchvision import transforms
    count = 274
    for data in data_loader:
        org_image0 = data['image']
        pointcloud = data['points']
        K = data['camera']
        B, C, H, W = org_image0.shape
        pt30 = pointcloud[:, :3, :]
        f0 = pointcloud[:, 3:, :]
        pt20 = geo.project3Dto2D(pt30, K)  # B, 2, N
        sort_data0, sort_id0 = torch.sort(pt30[:, 2, :], dim=-1, descending=True)
        new_image1 = geo.generateImage(pt20, f0, sort_id0, [H, W])
        new_image1 = transforms.ToPILImage()(new_image1[0].data.cpu()).convert('RGB')
        new_image1.save("../visualize/%d.png" % count)
        count = count + 1
        print(count)
        # ptc = data['points'][0]
        # ptc_np = ptc.numpy().transpose()
        # fx, fy, cx, cy = torch.split(data['camera'][0], 1)
        # x = ptc[0, :]
        # y = ptc[1, :]
        # z = ptc[2, :]
        #
        # u = x / z * fx + cx
        # v = y / z * fy + cy
        #
        # image_a = transforms.ToPILImage()(data['image'][0]).convert('RGB')
        # image_a.save("image_a.png")
        # draw = ImageDraw.Draw(image_a)
        # for i in range(x.shape[0]):
        #     draw.point((u[i], v[i]), fill='red')
        # image_a.save("image_b.png")
        # break



