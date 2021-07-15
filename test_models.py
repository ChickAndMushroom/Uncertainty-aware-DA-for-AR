import argparse
import time
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

from dataset import TSNDataSet
from models import CADAModel
from utils.utils import plot_confusion_matrix
import matplotlib.pyplot as plt

from colorama import init
from colorama import Fore, Back, Style
from tqdm import tqdm
from time import sleep
import matplotlib
matplotlib.use('TkAgg')
cudnn.enabled = False
init(autoreset=True)

# options
parser = argparse.ArgumentParser(description="Standard video-level testing")
parser.add_argument('class_file', type=str, default="classInd.txt")
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus'])
parser.add_argument('train_source_list', type=str)
parser.add_argument('train_target_list', type=str)
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--test_segments', type=int, default=5)
parser.add_argument('--add_fc', default=1, type=int, metavar='M', help='number of additional fc layers (excluding the last fc layer) (e.g. 0, 1, 2, ...)')
parser.add_argument('--fc_dim', type=int, default=512, help='dimension of added fc')
parser.add_argument('--baseline_type', type=str, default='frame', choices=['frame', 'video', 'tsn'])
parser.add_argument('--frame_aggregation', type=str, default='avgpool', choices=['avgpool', 'rnn', 'temconv', 'trn-m', 'none'], help='aggregation of frame features (none if baseline_type is not video)')
parser.add_argument('--dropout_i', type=float, default=0)
parser.add_argument('--dropout_v', type=float, default=0)

#------ RNN ------
parser.add_argument('--n_rnn', default=1, type=int, metavar='M',
                    help='number of RNN layers (e.g. 0, 1, 2, ...)')
parser.add_argument('--rnn_cell', type=str, default='LSTM', choices=['LSTM', 'GRU'])
parser.add_argument('--n_directions', type=int, default=1, choices=[1, 2],
                    help='(bi-) direction RNN')
parser.add_argument('--n_ts', type=int, default=5, help='number of temporal segments')

# ========================= DA Configs ==========================
parser.add_argument('--share_params', type=str, default='Y', choices=['Y', 'N'])
parser.add_argument('--use_bn', type=str, default='none', choices=['none', 'AdaBN', 'AutoDIAL'])
parser.add_argument('--use_attn_frame', type=str, default='none', choices=['none', 'TransAttn', 'general', 'DotProduct'], help='attention-mechanism for frames only')
parser.add_argument('--use_attn', type=str, default='none', choices=['none', 'TransAttn', 'general', 'DotProduct'], help='attention-mechanism')
parser.add_argument('--n_attn', type=int, default=1, help='number of discriminators for transferable attention')

# ========================= Monitor Configs ==========================
parser.add_argument('--top', default=[1, 3, 5], nargs='+', type=int, help='show top-N categories')
parser.add_argument('--verbose', default=False, action="store_true")

# ========================= Runtime Configs ==========================
parser.add_argument('--save_confusion', type=str, default=None)
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--save_attention', type=str, default=None)
parser.add_argument('--max_num', type=int, default=-1, help='number of videos to test')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--bS', default=2, help='batch size', type=int, required=False)
parser.add_argument('--bS_2', default=2, help='batch size', type=int, required=False)
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')

args = parser.parse_args()

class_names = [line.strip().split(' ', 1)[1] for line in open(args.class_file)]
num_class = len(class_names)

#=== Load the network ===#
print(Fore.CYAN + 'preparing the model......')
net = CADAModel(num_class, args.baseline_type, args.frame_aggregation, args.modality,
		train_segments=args.test_segments if args.baseline_type == 'video' else 1, val_segments=args.test_segments if args.baseline_type == 'video' else 1,
		base_model=args.arch, add_fc=args.add_fc, fc_dim=args.fc_dim, share_params=args.share_params,
		dropout_i=args.dropout_i, dropout_v=args.dropout_v, use_bn=args.use_bn, partial_bn=False,
		n_rnn=args.n_rnn, rnn_cell=args.rnn_cell, n_directions=args.n_directions, n_ts=args.n_ts,
		use_attn=args.use_attn, n_attn=args.n_attn, use_attn_frame=args.use_attn_frame,
		verbose=args.verbose)

checkpoint = torch.load(args.weights)

print("model epoch {} prec@1: {}".format(checkpoint['epoch'], checkpoint['prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
net.load_state_dict(checkpoint['state_dict'])

#=== Data loading ===#
print(Fore.CYAN + 'loading data......')

data_length = 1 if args.modality == "RGB" else 5
num_test = sum(1 for i in open(args.test_list))
num_source_train = sum(1 for i in open(args.train_source_list))
num_target_train = sum(1 for i in open(args.train_target_list))

source_set = TSNDataSet("", args.train_source_list, num_dataload=num_source_train, num_segments=args.test_segments,
						new_length=data_length, modality=args.modality,
						image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix+"{}_{:05d}.t7",
						test_mode=True,
						)
source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.bS, shuffle=False, num_workers=args.workers, pin_memory=True)

target_set = TSNDataSet("", args.train_target_list, num_dataload=num_target_train, num_segments=args.test_segments,
						new_length=data_length, modality=args.modality,
						image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
						test_mode=True,
						)
target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.bS_2, shuffle=False,  num_workers=args.workers, pin_memory=True)


data_set = TSNDataSet("", args.test_list, num_dataload=num_test, num_segments=args.test_segments,
	new_length=data_length, modality=args.modality,
	image_tmpl="img_{:05d}.t7" if args.modality in ['RGB', 'RGBDiff', 'RGBDiff2', 'RGBDiffplus'] else args.flow_prefix+"{}_{:05d}.t7",
	test_mode=True,
	)
data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.bS, shuffle=False, num_workers=args.workers, pin_memory=True)

data_gen = tqdm(data_loader)

st_loader = enumerate(zip(source_loader, target_loader))

#--- GPU processing ---#
net = torch.nn.DataParallel(net.cuda())
net.eval()
tsne = TSNE(n_components=2)

output = []
attn_values = torch.Tensor()
##########################################################
def myRemoveDummy(lc, batch_size_ori):
	lc = lc[:batch_size_ori]

	return lc

#############################################################
def eval_video(video_data):
	i, data, label = video_data

	data = data.cuda()
	label = label.cuda(non_blocking=True) # pytorch 0.4.X

	num_crop = 1 

	# e.g.
	# data.shape = [1,sample # x 2048]
	# data.view(-1, length, data.size(2), data.size(3)).shape = [sample #,2048]

	out, _, _, _, _, _,  _, _,_,_ = net(data, data, [0, 0, 0], 0, is_train=False, reverse=False)
	out = nn.Softmax(dim=1)(out).topk(max(args.top))
	prob = out[0].data.cpu().numpy().copy() # rst.shape = [sample #, top class #]
	pred_labels = out[1].data.cpu().numpy().copy() # rst.shape = [sample #, top class #]

	if args.baseline_type == 'video':
		prob_video = prob.reshape((data.size(0), num_crop, max(args.top))).mean(axis=1).reshape(
		(data.size(0), max(args.top)))
	else:
		prob_video = prob.reshape((data.size(0), num_crop, args.test_segments, max(args.top))).mean(axis=1).reshape(
		(data.size(0), args.test_segments, max(args.top)))
		prob_video = np.mean(prob_video, axis=1)


	return i, prob_video, pred_labels, label.cpu().numpy()
#############################################################
#=== t-SNE ===#
print(Fore.CYAN + 'start t-SNE......')

source_feat_before_adapted = []
target_feat_before_adapted = []
source_feat_after_adapted = []
target_feat_after_adapted = []

for i, ((source_data, source_label),(target_data, target_label)) in st_loader:
	# collect data before adapted
	source_x = (source_data.reshape(source_data.shape[0], source_data.shape[1] * source_data.shape[2]))
	target_x = (target_data.reshape(target_data.shape[0], target_data.shape[1] * target_data.shape[2]))	
	if i==0:
		source_feat_before_adapted = source_x
		target_feat_before_adapted = target_x
	else:
		source_feat_before_adapted = torch.cat((source_feat_before_adapted,source_x))
		target_feat_before_adapted = torch.cat((target_feat_before_adapted,target_x))
	
	source_size_ori = source_data.size()  # original shape
	target_size_ori = target_data.size()  # original shape
	batch_source_ori = source_size_ori[0]
	batch_target_ori = target_size_ori[0]
	# add dummy tensors to keep the same batch size for each epoch (for the last epoch)
	if batch_source_ori < args.bS:
		source_data_dummy = torch.zeros(args.bS - batch_source_ori, source_size_ori[1], source_size_ori[2])
		source_data = torch.cat((source_data, source_data_dummy))
	if batch_target_ori < args.bS_2:
		target_data_dummy = torch.zeros(args.bS_2 - batch_target_ori, target_size_ori[1], target_size_ori[2])
		target_data = torch.cat((target_data, target_data_dummy))

	# add dummy tensors to make sure batch size can be divided by gpu #
	#if source_data.size(0) % gpu_count != 0:
	#	source_data_dummy = torch.zeros(gpu_count - source_data.size(0) % gpu_count, source_data.size(1), source_data.size(2))
	#	source_data = torch.cat((source_data, source_data_dummy))
	#if target_data.size(0) % gpu_count != 0:
	#	target_data_dummy = torch.zeros(gpu_count - target_data.size(0) % gpu_count, target_data.size(1), target_data.size(2))
	#	target_data = torch.cat((target_data, target_data_dummy))


	_, _, _, _, _, _,  _, _,adapted_source_feat, adapted_target_feat = net(source_data, target_data, [0, 0, 0], 0, is_train=False, reverse=False)
	# ignore dummy tensors
	adapted_source_feat = myRemoveDummy(adapted_source_feat, batch_source_ori)
	adapted_target_feat = myRemoveDummy(adapted_target_feat, batch_target_ori)
	
	# collect feature after adapted
	if i==0:
		source_feat_after_adapted = adapted_source_feat
		target_feat_after_adapted = adapted_target_feat
	else:
		source_feat_after_adapted = torch.cat((source_feat_after_adapted,adapted_source_feat))
		target_feat_after_adapted = torch.cat((target_feat_after_adapted,adapted_target_feat))
# tsne feature before adapted
tsne_source_before = tsne.fit_transform(source_feat_before_adapted)
tsne_target_before = tsne.fit_transform(target_feat_before_adapted)
plt.figure(1)
plt.scatter(tsne_source_before[:, 0], tsne_source_before[:, 1], color='b', s=1, label='source')
plt.scatter(tsne_target_before[:, 0], tsne_target_before[:, 1], color='r', s=1, label='target')
plt.title('features before adapted')
plt.show()
# tsne feature after adapted
tsne_source_after = tsne.fit_transform(source_feat_after_adapted.cpu().detach().numpy())
tsne_target_after = tsne.fit_transform(target_feat_after_adapted.cpu().detach().numpy())
plt.figure(2)
plt.scatter(tsne_source_after[:, 0], tsne_source_after[:, 1], color='b', s=1, label='source')
plt.scatter(tsne_target_after[:, 0], tsne_target_after[:, 1], color='r', s=1, label='target')
plt.title('features after adapted')
plt.show()
#############################################################
proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

count_correct_topK = [0 for i in range(len(args.top))]
count_total = 0
video_pred = [[] for i in range(max(args.top))]
video_labels = []

#=== Testing ===#
print(Fore.CYAN + 'start testing......')
for i, (data, label) in enumerate(data_gen):
	data_size_ori = data.size() # original shape
	if data_size_ori[0] < args.bS:
		data_dummy = torch.zeros(args.bS - data_size_ori[0], data_size_ori[1], data_size_ori[2])
		data = torch.cat((data, data_dummy))
		label_dummy = torch.zeros(args.bS - data_size_ori[0]).long()
		label = torch.cat((label, label_dummy))

	if i >= max_num:
		break
	rst = eval_video((i, data, label))

	# remove the dummy part
	probs = rst[1][:data_size_ori[0]] # rst[1].shape = [sample #, top class #]
	preds = rst[2][:data_size_ori[0]]
	labels = rst[3][:data_size_ori[0]]
	#attn = rst[4][:data_size_ori[0]]

	#attn_values = torch.cat((attn_values, attn))  # save the attention values

	# accumulate
	for j in range(len(args.top)):
		for k in range(args.top[j]):
			count_correct_topK[j] += ((preds[:,k] == labels) * 1).sum()
	count_total += (preds[:,0].shape)[0]
	acc_topK = [float(count_correct_topK[j]) / float(count_total) for j in range(len(args.top))]

	for k in range(max(args.top)):
		video_pred[k] += preds[:,k].tolist() # save the top-K prediction

	video_labels += labels.tolist()

	cnt_time = time.time() - proc_start_time
	line_print = '                                                            ' # leave a large space for the tqdm progess bar
	for j in range(len(args.top)):
		line_print += 'Pred@%d %f, ' % (args.top[j], acc_topK[j])

	line_print += 'average %f sec/video\r' % (float(cnt_time) / (i + 1) / args.bS)
	data_gen.set_description(line_print)



cf = [confusion_matrix(video_labels, video_pred[k], labels=list(range(num_class))) for k in range(max(args.top))]

plot_confusion_matrix(args.save_confusion+'.png', cf[0], classes=class_names, normalize=True,
					  title='Normalized confusion matrix')

#--- overall accuracy ---#
cls_cnt = cf[0].sum(axis=1)
cls_hit = np.array([np.diag(cf[i]) for i in range(max(args.top))])
cls_acc_topK = [cls_hit[:j].sum(axis=0) / cls_cnt for j in args.top]

if args.verbose:
	for i in range(len(cls_acc_topK[0])):
		line_print = ''
		for j in range(len(args.top)):
			line_print += str(cls_acc_topK[j][i]) + ' '
		print(line_print)

final_line = ''
for j in args.top:
	final_line += Fore.YELLOW + 'Pred@{:d} {:.02f}% '.format(j, np.sum(cls_hit[:j].sum(axis=0)) / np.sum(cls_cnt) * 100)
print(final_line)

if args.save_confusion:
	class_acc_file = open(args.save_confusion + '-top' + str(args.top) + '.txt', 'w')

	for i in range(len(cls_acc_topK[0])):
		line_print = ''
		for j in range(len(args.top)):
			line_print += str(cls_acc_topK[j][i]) + ' '
		class_acc_file.write(line_print + '\n')

	class_acc_file.close()


if args.save_scores is not None: 
	# reorder before saving
	name_list = [x.strip().split()[0] for x in open(args.test_list)]

	order_dict = {e:i for i, e in enumerate(sorted(name_list))}

	reorder_output = [None] * len(output)
	reorder_label = [None] * len(output)

	for i in range(len(output)):
		idx = order_dict[name_list[i]]
		reorder_output[idx] = output[i]
		reorder_label[idx] = video_labels[i]

	np.savez(args.save_scores, scores=reorder_output, labels=reorder_label)
	
