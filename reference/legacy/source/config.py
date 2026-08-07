import argparse

parser = argparse.ArgumentParser()

def get_config():
    opt = parser.parse_args()
    return opt

# Hardware options
parser.add_argument('--gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--cpu', action='store_true', help='use cpu only')


# data in/out and dataset
parser.add_argument('--dataset_path_tr', default='/home/uanazodo/my_envs/sagan1/dataset_sagan/data/train/', help='fixed trainset root path')
parser.add_argument('--dataset_path_ts', default='/home/uanazodo/my_envs/sagan1/dataset_sagan/data/val/', help='fixed trainset root path')
parser.add_argument('--save_path', default='/home/uanazodo/my_envs/SMART3D/save_smart3d', help='save path')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training (default: 1)')
parser.add_argument('--patch_size', type=int, default=32, help='generate patches of size  (default: 5)')
parser.add_argument('--samples_per_volume', type=int, default=216, help='Default number of patches to extract from each volume (default: 9)')
parser.add_argument('--max_length', type=int, default=432, help='Maximum number of patches that can be stored in the queue (default: 300)')
parser.add_argument('--ds_input_nc', type=int, default=1, help='input channel for downsizing (default: 1)')
parser.add_argument('--ind_epoch', type=int, default=150, help='Best validation model')
parser.add_argument('--ds_filt_num', type=int, default=1024, help='input channel for downsizing (default: 1)')
parser.add_argument('--root', default='/home/uanazodo/my_envs/SMART3D/Dataset/evalaute/', help='model evaluation root path')



# train
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate (default: 0.0001)')
parser.add_argument('--b1', type=float, default=0.5, help='betas coefficients b1  (default: 0.5)')
parser.add_argument('--b2', type=float, default=0.999, help='betas coefficients b2  (default: 0.999)')
parser.add_argument('--epochs', type=int, default=400, help='number of epochs to train (default: 200)')
parser.add_argument('--decay_epoch', type=int, default=100, help='number of epochs to train (default: 50)')
parser.add_argument('--n_critic', type=int, default=5, help='number of iterations of critic(discriminator) (default: 5)')
parser.add_argument('--checkpoint_interval', default=1, type=int, help='checkpoint_interval (default: 5)')

parser.add_argument('--target_size', type=int, default=256, help='Input shape (default: 256)')
parser.add_argument('--validation_split', type=float, default=0.25, help='validation_split (default: 0.25)')



opt = get_config()
