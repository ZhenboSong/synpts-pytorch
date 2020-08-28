def add_basics_config(parser):
    parser.add_argument('--root_dir', default='/home/song/Documents/data/invsfm_data', type=str, help='path to the dataset folder.')
    parser.add_argument('--val_data_dir', default='data/demon_valid.txt', type=str, help='path to the validation dataset file.')
    parser.add_argument('--train_data_dir', default='data/demon_train.txt', type=str, help='path to the train/test dataset file.')
    parser.add_argument('--test_data_dir', default='data/filenames/inv_test.txt', type=str, help='path to the train/test dataset file.')
    parser.add_argument('--output_directory', default='results', type=str, help='where save dispairities for tested images')
    parser.add_argument('--image_size', default=(256, 256), type=tuple, help='image size in the network, as type tuple (height, width)')
    parser.add_argument('--pointcloud_size', default=4096, type=int)
    parser.add_argument('--device', default='cuda:0', type=str, help='choose cpu or cuda:0 device"')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers in dataloader')
    parser.add_argument('--use_multiple_gpu', default=False, type=bool)


def add_network_config(parser):
    parser.add_argument('--model_name', default='demon')
    parser.add_argument('--mode', default='test', type=str, choices=['train', 'test'], help='mode: train or test (default: train)')
    parser.add_argument('--pre_trained', default='not_trained', type=str, choices=('total_trained', 'part_trained', 'not_trained'), help='Use weights of pretrained model')
    parser.add_argument('--dis_model', type=str, default='results/dis_demon_best.pth')
    parser.add_argument('--pte_model', type=str, default='results/demon_4096/pte_demon_best.pth')
    parser.add_argument('--imgd_model', type=str, default='results/demon_4096/imgd_demon_best.pth')
    parser.add_argument('--refine_model', type=str, default='results/demon_4096/gen_demon_best.pth')
    parser.add_argument('--use_refine', type=bool, default=True)
    parser.add_argument('--only_train_refine', type=bool, default=False)


def add_pointnet_config(parser):
    parser.add_argument('--points_channel', type=int, default=3)
    parser.add_argument('--use_xyz', type=bool, default=True)
    parser.add_argument('--use_bn', type=bool, default=True)
    parser.add_argument('--sa_npoints', type=list, default=[2048, 512, 128, 32], help='uniform number of groups using furthest point sampling')
    parser.add_argument('--sa_mlps', type=list, default=[[[16, 16, 32], [32, 32, 64]],
                                                          [[64, 64, 128], [64, 96, 128]],
                                                          [[128, 196, 256], [128, 196, 256]],
                                                          [[256, 256, 512], [256, 384, 512]]])
    parser.add_argument('--sa_nsample', type=list, default=[[16, 32], [16, 32], [16, 32], [16, 32]], help='two multi scale sampling for each layer')
    parser.add_argument('--sa_radius', type=list, default=[[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]], help='two radiuses for each layer')
    parser.add_argument('--fp_mlps', type=list, default=[[128, 128], [256, 256], [512, 512], [512, 512]])


def add_loss_config(parser):
    parser.add_argument('--gan_loss', default='nsgan', type=str, choices=('nsgan', 'lsgan', 'hinge'), help='Choose a GAN loss type.\n')
    parser.add_argument('--l1_loss_w', default=1)
    parser.add_argument('--style_loss_w', default=250)
    parser.add_argument('--content_loss_w', default=0.1)
    parser.add_argument('--adversarial_loss_w', default=0.1)


def add_train_config(parser):
    parser.add_argument('--epochs', default=50, help='number of total epochs to run')
    parser.add_argument('--learning_rate', default=1e-4, help='initial learning rate (default: 1e-4)')
    parser.add_argument('--dis2gen_lr', default=0.1, help="discriminator/generator learning rate ratio")
    parser.add_argument('--beta1', default=0.0, help="adam optimizer beta1")
    parser.add_argument('--beta2', default=0.9, help="adam optimizer beta2")
    parser.add_argument('--batch_size', default=2, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--adjust_lr', default=True, help='apply learning rate decay or not (default: True)')
    parser.add_argument('--lr_decay_ratio', default=0.5, type=float, help='lr decay ratio (default:0.5)')
    parser.add_argument('--lr_decay_epochs', default=10, type=int, help='lr decay epochs')



