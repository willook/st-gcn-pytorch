from pathlib import Path
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--data_path',  type=str, default='./dataset/ntu_rgb')
	parser.add_argument('--model_path',  type=str, default='./models')
	parser.add_argument('--use_2d',  action='store_true')
	parser.add_argument('--recons', type=str2bool, default=True)
	parser.add_argument('--triplet', type=str2bool, default=True)
	
	parser.add_argument('--mode',  type=str, default='train')
	parser.add_argument('--batch_size',  type=int, default=1)
	parser.add_argument('--learning_rate',type=int, default=0.0005)
	parser.add_argument('--beta1',type=int, default=0.5)
	parser.add_argument('--beta2',type=int, default=0.99)
	parser.add_argument('--dropout_rate',type=int, default=0.0)
	parser.add_argument('--weight_decay',type=int, default=0.0)

	# train process
	parser.add_argument('--num_epochs',type=int, default=100)
	parser.add_argument('--start_epoch',type=int, default=0)
	parser.add_argument('--test_epoch',type=int, default=30)
	parser.add_argument('--val_step',type=int, default=2)

	# model params
	parser.add_argument('--num_classes',type=int, default=121)
	parser.add_argument('--feat_dims',type=int, default=13)
	
	args = parser.parse_args()

	dataset_path = Path(args.data_path)
	args.train_path = dataset_path / "train.pkl"
	args.test_path = dataset_path / "test.pkl"
	args.valid_path = dataset_path / "valid.pkl"
	
	return args
