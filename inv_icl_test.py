import argparse, time
import torch
import collections
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from data.inv_data import InvLoader
from data.icl_dso_data import ICLDSOLoader
import config
from models.point2image import Point2ImageRefine


class Model:
    def __init__(self, args):
        self.args = args

        # Set up model
        self.device = args.device
        self.img_size = args.image_size

        self.model = Point2ImageRefine(args)

        self.model = self.model.to(self.device)
        if args.use_multiple_gpu:
            self.model = torch.nn.DataParallel(self.model)

        self.n_img, self.loader = self.dataloader(args.test_data_dir)
        args.augment_parameters = None
        args.do_augmentation = False
        args.batch_size = 1

        self.model.load()

        self.output_directory = args.output_directory

        if 'cuda' in self.device:
            torch.cuda.synchronize()

    def test_process(self):
        count = 0
        endtime = time.time()
        for data in self.loader:
            data = self.to_device(data, self.device)
            real_img = data['image'][0]
            ptc = data['points']
            calib = data['camera']

            T = torch.eye(4, device=calib.device, dtype=calib.dtype).unsqueeze(0).repeat(self.args.batch_size, 1, 1)
            with torch.no_grad():
                outputs, image_tensor = self.model(ptc, T, calib)

            # image = transforms.ToPILImage()(image_tensor[0].data.cpu()).convert('RGB')
            # image.save("results/ptc/%d.png" % count)
            image = transforms.ToPILImage()(outputs[1][0].data.cpu()).convert('RGB')
            image.save("results/fine/%d.png" % count)
            # image = transforms.ToPILImage()(outputs[0][0].data.cpu()).convert('RGB')
            # image.save("results/coarse/%d.png" % count)
            image = transforms.ToPILImage()(real_img[0].data.cpu()).convert('RGB')
            image.save("results/real/%d.png" % count)

            count = count + 1
            print('finish:', count,
                  'time:', time.time() - endtime, 's')

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

    def dataloader(self, file_dir):
        if self.args.dataset == 'invsfm':
            dataset = InvLoader(self.args.root_dir, file_dir, self.args.image_size, self.args.pointcloud_size)
        else:
            dataset = ICLDSOLoader(self.args.root_dir, file_dir, self.args.image_size, self.args.pointcloud_size)
        n_img = len(dataset)

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
    parser.add_argument('--dataset', default='icl', type=str, choices=('icl', 'invsfm'))
    args = parser.parse_args()

    model_test = Model(args)
    model_test.test_process()


if __name__ == '__main__':
    main()
