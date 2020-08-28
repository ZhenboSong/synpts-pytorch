import argparse, time
import torch
import collections
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import shutil
from torchvision.transforms import transforms

from models.loss import PSNR
import models.geometry as geo
from data.demon_data import DeMonLoader
import config
from models.point2image import Point2ImageModel, Point2ImageRefine


class Model:
    def __init__(self, args):
        self.args = args

        # Set up model
        self.device = args.device
        self.img_size = args.image_size
        if self.args.use_refine:
            self.model = Point2ImageRefine(args)
        else:
            self.model = Point2ImageModel(args)
        self.model = self.model.to(self.device)
        if args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)

        if args.mode == 'train':
            self.val_n_img, self.val_loader = self.dataloader('validation', args.val_data_dir)
            self.train_n_img, self.train_loader = self.dataloader('train', args.train_data_dir)
            # logger
            date_name = datetime.now().strftime('%b%d_%H-%M')
            logfile_name = '_'.join([date_name, args.model_name])
            logfile_name = os.path.join(args.output_directory, 'logs', logfile_name)
            if os.path.isdir(logfile_name):
                shutil.rmtree(logfile_name)
            os.makedirs(logfile_name)
            self.logger = SummaryWriter(logfile_name)
            self.psnr = PSNR(255.0).to(self.device)
        else:
            self.n_img, self.loader = self.dataloader('test', args.test_data_dir)
            args.augment_parameters = None
            args.do_augmentation = False
            args.batch_size = 1

        if args.pre_trained == 'total_trained':
            self.model.load()
        elif args.pre_trained == 'part_trained':
            self.model.load_part()

        self.output_directory = args.output_directory

        if 'cuda' in self.device:
            torch.cuda.synchronize()

    def gen_new_image(self, point3d, feature2d, K, transform=None):
        if transform is not None:
            pt3_trans = geo.transform3Dto3D(point3d, transform)  # B, 3, N
        else:
            pt3_trans = point3d
        pt2_trans = geo.project3Dto2D(pt3_trans, K)  # B, 2, N
        # sort depth
        sort_data, sort_id = torch.sort(pt3_trans[:, 2, :], dim=-1, descending=True)
        new_image = geo.generateImage(pt2_trans, feature2d, sort_id, self.img_size)

        return new_image

    def validation(self, epoch):
        psnr_val_loss = 0.0
        mae_val_loss = 0.0
        count = 0
        endtime = time.time()
        for data in self.val_loader:
            data = self.to_device(data, self.device)
            real_img0 = data['image'][0]
            ptc = data['points']
            pose = data['warp']
            calib = data['camera']

            B, _, _ = pose.shape
            T = torch.eye(4, device=pose.device, dtype=pose.dtype).unsqueeze(0).repeat(B, 1, 1)
            with torch.no_grad():
                outputs, image_tensor = self.model(ptc, T, calib)
            if self.args.use_refine:
                coarse_out, fine_out = outputs
            else:
                fine_out = outputs

            # metrics
            psnr = self.psnr(self.postprocess(real_img0), self.postprocess(fine_out))
            mae = torch.sum(torch.abs(real_img0 - fine_out))
            psnr_val_loss += psnr.item()
            mae_val_loss += mae.item()

            count += 1
            if count % 100 == 0:
                print("validation %d / %d finished," % (count, int(self.val_n_img/self.args.batch_size)),
                      'psnr_loss:', psnr_val_loss/(count * self.args.batch_size),
                      'mae_loss:', mae_val_loss/(count * self.args.batch_size),
                      "time:", time.time() - endtime, "s")
                self.logger.add_image('valid_input', image_tensor[0][0, :3, :, :],
                                      epoch*int(self.val_n_img/self.args.batch_size) + count)
                self.logger.add_image('valid_real', real_img0[0],
                                      epoch * int(self.val_n_img / self.args.batch_size) + count)

                if self.args.use_refine:
                    self.logger.add_image('valid_coarse', coarse_out[0],
                                          epoch * int(self.val_n_img / self.args.batch_size) + count)
                self.logger.add_image('valid_output', fine_out[0],
                                      epoch*int(self.val_n_img/self.args.batch_size) + count)

        psnr_val_loss /= self.val_n_img
        mae_val_loss /= self.val_n_img

        print('psnr_loss:', psnr_val_loss,
              'mae_loss:', mae_val_loss,
              'time:', time.time() - endtime, 's')
        self.logger.add_scalar('psnr_val', psnr_val_loss, epoch)
        self.logger.add_scalar('mae_val', mae_val_loss, epoch)

        return mae_val_loss

    def train(self, epoch):
        endtime = time.time()
        psnr_train_loss = 0.0
        mae_train_loss = 0.0
        self.model.train()
        count = 0
        for data in self.train_loader:
            data = self.to_device(data, self.device)
            real_img0 = data['image'][0]
            real_img1 = data['image'][1]
            ptc = data['points']
            pose = data['warp']
            calib = data['camera']

            random_seed = np.random.uniform(-1., 1., 1)
            if random_seed > 0:
                B, _, _ = pose.shape
                T = torch.eye(4, device=pose.device, dtype=pose.dtype).unsqueeze(0).repeat(B, 1, 1)

                outputs, gen_loss, dis_loss, logs, image_tensor = self.model.process(ptc, T, calib, real_img0)
                if self.args.use_refine:
                    coarse_out, fine_out = outputs
                else:
                    fine_out = outputs
                psnr = self.psnr(self.postprocess(real_img0), self.postprocess(fine_out))
                mae = torch.sum(torch.abs(real_img0 - fine_out))
                psnr_train_loss += psnr.item()
                mae_train_loss += mae.item()
                real_img = real_img0
            else:
                B, _, _ = pose.shape
                outputs, gen_loss, dis_loss, logs, image_tensor = self.model.process(ptc, pose, calib, real_img0)
                if self.args.use_refine:
                    coarse_out, fine_out = outputs
                else:
                    fine_out = outputs
                psnr = self.psnr(self.postprocess(real_img1), self.postprocess(fine_out))
                mae = torch.sum(torch.abs(real_img1 - fine_out))
                psnr_train_loss += psnr.item()
                mae_train_loss += mae.item()
                real_img = real_img1

            self.model.backward(gen_loss, dis_loss)

            count += 1
            if count % 3000 == 0:
                print("train %d / %d finished," % (count, int(self.train_n_img/self.args.batch_size)),
                      "psnr_train_loss %lf" % (psnr_train_loss/(count * self.args.batch_size)),
                      "mae_train_loss %lf" % (mae_train_loss/(count * self.args.batch_size)),
                      "time:", time.time() - endtime, "s")
                self.logger.add_image('train_input', image_tensor[0][0, :3, :, :], epoch*self.train_n_img/self.args.batch_size + count)
                self.logger.add_image('train_real', real_img[0], epoch * self.train_n_img / self.args.batch_size + count)
                if self.args.use_refine:
                    self.logger.add_image('train_coarse', coarse_out[0], epoch*self.train_n_img/self.args.batch_size + count)
                self.logger.add_image('train_output', fine_out[0], epoch*self.train_n_img/self.args.batch_size + count)
                for key, num in logs[1].items():
                    self.logger.add_scalar(key, num, epoch*self.train_n_img/self.args.batch_size + count)

        psnr_train_loss /= count * self.args.batch_size
        mae_train_loss /= count * self.args.batch_size

        endtime = time.time() - endtime
        print("mae_train_loss:", mae_train_loss, "time:", endtime, "s")

        if self.args.adjust_lr:
            self.model.lr_step()

    def train_process(self):
        best_result = self.validation(0)
        for epoch in range(self.args.epochs):
            self.train(epoch)
            valid_loss = self.validation(epoch)

            pte_path = self.args.output_directory + '/pte_' + self.args.model_name
            imgd_path = self.args.output_directory + '/imgd_' + self.args.model_name
            dis_path = self.args.output_directory + '/dis_' + self.args.model_name
            gen_path = self.args.output_directory + '/gen_' + self.args.model_name
            if self.args.use_refine:
                self.model.save(pte_path+"_last.pth", imgd_path+"_last.pth", gen_path + "_last.pth", dis_path + "_last.pth")
            else:
                self.model.save(pte_path+"_last.pth", imgd_path+"_last.pth", dis_path+"_last.pth")
            if valid_loss < best_result:
                if self.args.use_refine:
                    self.model.save(pte_path+"_best.pth", imgd_path+"_best.pth", gen_path+"_best.pth", dis_path+"_best.pth")
                else:
                    self.model.save(pte_path+"_best.pth", imgd_path+"_best.pth", dis_path+"_best.pth")

                best_result = valid_loss
                print('Model_saved')

    def test_process(self):
        count = 0
        endtime = time.time()
        for data in self.loader:
            data = self.to_device(data, self.device)
            real_img0, real_img1 = data['image']
            ptc = data['points']
            pose = data['warp']
            calib = data['camera']

            B, C, H, W = real_img0.shape
            pt30 = ptc[:, :3, :]
            f0 = ptc[:, 3:, :].clone()
            pt20 = geo.project3Dto2D(pt30, calib)  # B, 2, N
            sort_data0, sort_id0 = torch.sort(pt30[:, 2, :], dim=-1, descending=True)
            new_image0 = geo.generateImage(pt20, f0, sort_id0, [H, W])
            pt31 = geo.transform3Dto3D(pt30, pose)
            pt21 = geo.project3Dto2D(pt31, calib)
            sort_data1, sort_id1 = torch.sort(pt31[:, 2, :], dim=-1, descending=True)
            new_image1 = geo.generateImage(pt21, f0, sort_id1, [H, W])
            T = torch.eye(4, device=pose.device, dtype=pose.dtype).unsqueeze(0).repeat(B, 1, 1)
            with torch.no_grad():
                outputs, image_tensor = self.model(ptc, T, calib)

            image = transforms.ToPILImage()(outputs[1][0].data.cpu()).convert('RGB')
            image.save("visualize/demon/fine/%d.png" % count)
            image = transforms.ToPILImage()(outputs[0][0].data.cpu()).convert('RGB')
            image.save("visualize/demon/coarse/%d.png" % count)
            image = transforms.ToPILImage()(real_img1[0].data.cpu()).convert('RGB')
            image.save("visualize/demon/real/%d.png" % count)
            image = transforms.ToPILImage()(new_image0[0].data.cpu()).convert('RGB')
            image.save("visualize/demon/ptc/%d.png" % count)
            count = count + 1
            print('finish:', count,
                  'time:', time.time() - endtime, 's')

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def to_device(self, data, device):
        if torch.is_tensor(data):
            return data.to(device=device)
        elif isinstance(data, str):
            return data
        elif isinstance(data, collections.Mapping):
            return {k: self.to_device(sample, device=device) for k, sample in data.items()}
        elif isinstance(data, collections.Sequence):
            return [self.to_device(sample, device=device) for sample in data]
        else:
            raise TypeError(f"Input must contain tensor, dict or list, found {type(data)}")

    def dataloader(self, mode, file_dir):
        dataset = DeMonLoader(self.args.root_dir, file_dir, self.args.image_size, self.args.pointcloud_size)
        n_img = len(dataset)
        if mode == 'train':
            print('Use a training dataset with', n_img, 'images')
            loader = DataLoader(dataset, batch_size=self.args.batch_size,
                                shuffle=True, num_workers=self.args.num_workers,
                                pin_memory=True)
        elif mode == 'validation':
            print('Use a validation dataset with', n_img, 'images')
            loader = DataLoader(dataset, batch_size=self.args.batch_size,
                                shuffle=True, num_workers=self.args.num_workers,
                                pin_memory=True)
        else:
            print('Use a testing dataset with', n_img, 'images')
            loader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=self.args.num_workers,
                                pin_memory=True)
        return n_img, loader


def main():
    parser = argparse.ArgumentParser(description='Training the network')

    config.add_basics_config(parser)
    config.add_network_config(parser)
    config.add_pointnet_config(parser)
    config.add_loss_config(parser)
    config.add_train_config(parser)

    args = parser.parse_args()

    if args.mode == 'train':
        model = Model(args)
        model.train_process()
    elif args.mode == 'test':
        model_test = Model(args)
        model_test.test_process()


if __name__ == '__main__':
    main()
