from torch import nn

from torch.nn.init import *
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import TRNmodule
import math

from colorama import init
from colorama import Fore, Back, Style

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

init(autoreset=True)

# definition of Gradient Reversal Layer
class GradReverse(Function):
	@staticmethod
	def forward(ctx, x, beta):
		ctx.beta = beta
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		grad_input = grad_output.neg() * ctx.beta
		return grad_input, None

# definition of Gradient Scaling Layer
class GradScale(Function):
	@staticmethod
	def forward(ctx, x, beta):
		ctx.beta = beta
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		grad_input = grad_output * ctx.beta
		return grad_input, None

# definition of Temporal-ConvNet Layer
class TCL(nn.Module):
	def __init__(self, conv_size, dim):
		super(TCL, self).__init__()

		self.conv2d = nn.Conv2d(dim, dim, kernel_size=(conv_size,1), padding=(conv_size//2,0))

		# initialization
		kaiming_normal_(self.conv2d.weight)

	def	forward(self, x):
		x = self.conv2d(x)

		return x



class CADAModel(nn.Module):
	def __init__(self, num_class, baseline_type, frame_aggregation, modality,
				train_segments=5, val_segments=5,
				base_model='resnet101', path_pretrained='', new_length=None,
				before_softmax=True,
				dropout_i=0.5, dropout_v=0.5, use_bn='none', ens_DA='none',
				crop_num=1, partial_bn=True, verbose=True, add_fc=1, fc_dim=1024,
				n_rnn=1, rnn_cell='LSTM', n_directions=1, n_ts=5,
				use_attn='TransAttn', n_attn=1, use_attn_frame='none',
				share_params='Y'):
		super(CADAModel, self).__init__()
		self.modality = modality
		self.train_segments = train_segments
		self.val_segments = val_segments
		self.baseline_type = baseline_type
		self.frame_aggregation = frame_aggregation
		self.reshape = True
		self.before_softmax = before_softmax
		self.dropout_rate_i = dropout_i
		self.dropout_rate_v = dropout_v
		self.use_bn = use_bn
		self.ens_DA = ens_DA
		self.crop_num = crop_num
		self.add_fc = add_fc
		self.fc_dim = fc_dim
		self.share_params = share_params

		# RNN
		self.n_layers = n_rnn
		self.rnn_cell = rnn_cell
		self.n_directions = n_directions
		self.n_ts = n_ts # temporal segment

		# Attention
		self.use_attn = use_attn 
		self.n_attn = n_attn
		self.use_attn_frame = use_attn_frame

		if new_length is None:
			self.new_length = 1 if modality == "RGB" else 5
		else:
			self.new_length = new_length

		if verbose:
			print(("""
				Initializing TSN with base model: {}.
				TSN Configurations:
				input_modality:     {}
				num_segments:       {}
				new_length:         {}
				""".format(base_model, self.modality, self.train_segments, self.new_length)))

		self._prepare_DA(num_class, base_model)

		self.softmax = nn.Softmax(dim = -1)

	def _prepare_DA(self, num_class, base_model): # convert the model to DA framework
		if base_model == 'c3d': # C3D mode: in construction...
			from C3D_model import C3D
			model_test = C3D()
			self.feature_dim = model_test.fc7.in_features
		else:
			model_test = getattr(torchvision.models, base_model)(True) # model_test is only used for getting the dim #
			self.feature_dim = model_test.fc.in_features

		std = 0.001
		feat_shared_dim = min(self.fc_dim, self.feature_dim) if self.add_fc > 0 and self.fc_dim > 0 else self.feature_dim
		feat_frame_dim = feat_shared_dim
		feat_last_dim = 128 #here

		self.relu = nn.ReLU(inplace=True)
		self.softplus = nn.Softplus()
		self.dropout_i = nn.Dropout(p=self.dropout_rate_i)
		self.dropout_v = nn.Dropout(p=self.dropout_rate_v)

		#------ frame-level layers (shared layers + source layers + domain layers) ------#
		if self.add_fc < 1:
			raise ValueError(Back.RED + 'add at least one fc layer')

		# 1. shared feature layers
		self.fc_feature_shared_source = nn.Linear(self.feature_dim, feat_shared_dim)
		normal_(self.fc_feature_shared_source.weight, 0, std)
		constant_(self.fc_feature_shared_source.bias, 0)

		if self.add_fc > 1:
			self.fc_feature_shared_2_source = nn.Linear(feat_shared_dim, feat_shared_dim)
			normal_(self.fc_feature_shared_2_source.weight, 0, std)
			constant_(self.fc_feature_shared_2_source.bias, 0)

		if self.add_fc > 2:
			self.fc_feature_shared_3_source = nn.Linear(feat_shared_dim, feat_shared_dim)
			normal_(self.fc_feature_shared_3_source.weight, 0, std)
			constant_(self.fc_feature_shared_3_source.bias, 0)

		# 2. frame-level feature layers
		self.fc_feature_source = nn.Linear(feat_shared_dim, feat_frame_dim)
		normal_(self.fc_feature_source.weight, 0, std)
		constant_(self.fc_feature_source.bias, 0)

		# 3. domain feature layers (frame-level) # used
		self.fc_feature_domain = nn.Linear(feat_shared_dim, feat_frame_dim) #Linear(512,512)
		normal_(self.fc_feature_domain.weight, 0, std)
		constant_(self.fc_feature_domain.bias, 0)
		
		self.fc_feature_domain2 = nn.Linear(feat_frame_dim, feat_last_dim) #Linear(512,128)
		normal_(self.fc_feature_domain2.weight, 0, std)
		constant_(self.fc_feature_domain2.bias, 0)

		# 4. classifiers (frame-level)
		self.fc_classifier_source = nn.Linear(feat_frame_dim, num_class)
		normal_(self.fc_classifier_source.weight, 0, std)
		constant_(self.fc_classifier_source.bias, 0)

		self.fc_classifier_domain = nn.Linear(feat_last_dim, 2)
		normal_(self.fc_classifier_domain.weight, 0, std)
		constant_(self.fc_classifier_domain.bias, 0)

		# variance computation layer
		self.variance_domain = nn.Linear(feat_last_dim, 1)
		normal_(self.variance_domain.weight, 0, std)
		constant_(self.variance_domain.bias, 0)

		if self.share_params == 'N':
			self.fc_feature_shared_target = nn.Linear(self.feature_dim, feat_shared_dim)
			normal_(self.fc_feature_shared_target.weight, 0, std)
			constant_(self.fc_feature_shared_target.bias, 0)
			if self.add_fc > 1:
				self.fc_feature_shared_2_target = nn.Linear(feat_shared_dim, feat_shared_dim)
				normal_(self.fc_feature_shared_2_target.weight, 0, std)
				constant_(self.fc_feature_shared_2_target.bias, 0)
			if self.add_fc > 2:
				self.fc_feature_shared_3_target = nn.Linear(feat_shared_dim, feat_shared_dim)
				normal_(self.fc_feature_shared_3_target.weight, 0, std)
				constant_(self.fc_feature_shared_3_target.bias, 0)

			self.fc_feature_target = nn.Linear(feat_shared_dim, feat_frame_dim)
			normal_(self.fc_feature_target.weight, 0, std)
			constant_(self.fc_feature_target.bias, 0)
			self.fc_classifier_target = nn.Linear(feat_frame_dim, num_class)
			normal_(self.fc_classifier_target.weight, 0, std)
			constant_(self.fc_classifier_target.bias, 0)


		#------ aggregate frame-based features (frame feature --> video feature) ------#
		if self.frame_aggregation == 'rnn': # 2. rnn
			self.hidden_dim = feat_frame_dim
			if self.rnn_cell == 'LSTM':
				self.rnn = nn.LSTM(feat_frame_dim, self.hidden_dim//self.n_directions, self.n_layers, batch_first=True, bidirectional=bool(int(self.n_directions/2)))
			elif self.rnn_cell == 'GRU':
				self.rnn = nn.GRU(feat_frame_dim, self.hidden_dim//self.n_directions, self.n_layers, batch_first=True, bidirectional=bool(int(self.n_directions/2)))

			# initialization
			for p in range(self.n_layers):
				kaiming_normal_(self.rnn.all_weights[p][0])
				kaiming_normal_(self.rnn.all_weights[p][1])

			self.bn_before_rnn = nn.BatchNorm2d(1)
			self.bn_after_rnn = nn.BatchNorm2d(1)

		elif self.frame_aggregation == 'trn': # 4. TRN (ECCV 2018) ==> fix segment # for both train/val
			self.num_bottleneck = 512
			self.TRN = TRNmodule.RelationModule(feat_shared_dim, self.num_bottleneck, self.train_segments)
			self.bn_trn_S = nn.BatchNorm1d(self.num_bottleneck)
			self.bn_trn_T = nn.BatchNorm1d(self.num_bottleneck)
		elif self.frame_aggregation == 'trn-m':  # 4. TRN (ECCV 2018) ==> fix segment # for both train/val
			self.num_bottleneck = 256
			self.TRN = TRNmodule.RelationModuleMultiScale(feat_shared_dim, self.num_bottleneck, self.train_segments)
			self.bn_trn_S = nn.BatchNorm1d(self.num_bottleneck)
			self.bn_trn_T = nn.BatchNorm1d(self.num_bottleneck)
		#------- calculate the dim of video feature
		if self.frame_aggregation == 'avgpool': # 1. avgpool
			feat_aggregated_dim = feat_shared_dim
		if 'trn' in self.frame_aggregation : # 4. trn
			feat_aggregated_dim = self.num_bottleneck
		elif self.frame_aggregation == 'rnn': # 2. rnn
			feat_aggregated_dim = self.hidden_dim

		
		feat_video_dim = feat_aggregated_dim

		# 1. source feature layers (video-level)
		self.fc_feature_video_source = nn.Linear(feat_aggregated_dim, feat_video_dim)
		normal_(self.fc_feature_video_source.weight, 0, std)
		constant_(self.fc_feature_video_source.bias, 0)

		self.fc_feature_video_source_2 = nn.Linear(feat_video_dim, feat_video_dim)
		normal_(self.fc_feature_video_source_2.weight, 0, std)
		constant_(self.fc_feature_video_source_2.bias, 0)

		# 2. domain feature layers (video-level)
		self.fc_feature_domain_video = nn.Linear(feat_aggregated_dim, feat_video_dim)
		normal_(self.fc_feature_domain_video.weight, 0, std)
		constant_(self.fc_feature_domain_video.bias, 0)

		# 3. classifiers (video-level)
		self.fc_classifier_video_source = nn.Linear(feat_video_dim, num_class)
		normal_(self.fc_classifier_video_source.weight, 0, std)
		constant_(self.fc_classifier_video_source.bias, 0)

		# feature fc for video-level
		self.fc_feature_class_video = nn.Linear(feat_video_dim,feat_video_dim)
		normal_(self.fc_feature_class_video.weight, 0, std)
		constant_(self.fc_feature_class_video.bias, 0)

		# variance computation layer
		self.variance_class = nn.Linear(feat_video_dim, 1)
		normal_(self.variance_class.weight, 0, std)
		constant_(self.variance_class.bias, 0)

		self.fc_classifier_domain_video = nn.Linear(feat_video_dim, 2)
		normal_(self.fc_classifier_domain_video.weight, 0, std)
		constant_(self.fc_classifier_domain_video.bias, 0)


		# domain classifier for TRN-M
		if self.frame_aggregation == 'trn-m':
			self.relation_domain_classifier_all = nn.ModuleList()
			for i in range(self.train_segments-1):
				relation_domain_classifier = nn.Sequential(
					nn.Linear(feat_aggregated_dim, feat_video_dim),
					nn.ReLU(),
					nn.Linear(feat_video_dim, 2)
				)
				self.relation_domain_classifier_all += [relation_domain_classifier]
		# BN for the above layers

		# ------ attention mechanism ------#
		# conventional attention

	def get_trans_attn(self, pred_domain):
		softmax = nn.Softmax(dim=1)
		logsoftmax = nn.LogSoftmax(dim=1)
		entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), 1)
		weights = 1 - entropy

		return weights
	#def get_general_attn(self, feat):

	#def get_attn_feat_frame(self, feat_fc, pred_domain): # not used for now

	#def get_attn_feat_relation(self, feat_fc, pred_domain, num_segments):
	def get_attn_feat_relation(self, feat_fc, pred_domain, num_segments):
		if self.use_attn == 'TransAttn':
			weights_attn = self.get_trans_attn(pred_domain)
		elif self.use_attn == 'general':
			weights_attn = self.get_general_attn(feat_fc)

		weights_attn = weights_attn.view(-1, num_segments-1, 1).repeat(1,1,feat_fc.size()[-1]) # reshape & repeat weights (e.g. 16 x 4 x 256)
		feat_fc_attn = (weights_attn+1) * feat_fc

		return feat_fc_attn, weights_attn[:,:,0]

	def aggregate_frames(self, feat_fc, num_segments,pred_domain):
		feat_fc_video = None
		if self.frame_aggregation == 'rnn':
			# 2. RNN
			feat_fc_video = feat_fc.view((-1, num_segments) + feat_fc.size()[-1:])  # reshape for RNN

			# temporal segments and pooling
			len_ts = round(num_segments/self.n_ts)
			num_extra_f = len_ts*self.n_ts-num_segments
			if num_extra_f < 0: # can remove last frame-level features
				feat_fc_video = feat_fc_video[:, :len_ts * self.n_ts, :]  # make the temporal length can be divided by n_ts (16 x 25 x 512 --> 16 x 24 x 512)
			elif num_extra_f > 0: # need to repeat last frame-level features
				feat_fc_video = torch.cat((feat_fc_video, feat_fc_video[:,-1:,:].repeat(1,num_extra_f,1)), 1) # make the temporal length can be divided by n_ts (16 x 5 x 512 --> 16 x 6 x 512)

			feat_fc_video = feat_fc_video.view(
				(-1, self.n_ts, len_ts) + feat_fc_video.size()[2:])  # 16 x 6 x 512 --> 16 x 3 x 2 x 512
			feat_fc_video = nn.MaxPool2d(kernel_size=(len_ts, 1))(
				feat_fc_video)  # 16 x 3 x 2 x 512 --> 16 x 3 x 1 x 512
			feat_fc_video = feat_fc_video.squeeze(2)  # 16 x 3 x 1 x 512 --> 16 x 3 x 512

			hidden_temp = torch.zeros(self.n_layers * self.n_directions, feat_fc_video.size(0),
									  self.hidden_dim // self.n_directions).cuda()

			if self.rnn_cell == 'LSTM':
				hidden_init = (hidden_temp, hidden_temp)
			elif self.rnn_cell == 'GRU':
				hidden_init = hidden_temp

			self.rnn.flatten_parameters()
			feat_fc_video, hidden_final = self.rnn(feat_fc_video, hidden_init)  # e.g. 16 x 25 x 512

			# get the last feature vector
			feat_fc_video = feat_fc_video[:, -1, :]

		else:
			# 1. averaging
			feat_fc_video = feat_fc.view((-1, 1, num_segments) + feat_fc.size()[-1:])  # reshape based on the segments (e.g. 16 x 1 x 5 x 512)
			#if self.use_attn == 'TransAttn': # get the attention weighting
			#	weights_attn = self.get_trans_attn(pred_domain)
			#	weights_attn = weights_attn.view(-1, 1, num_segments,1).repeat(1,1,1,feat_fc.size()[-1]) # reshape & repeat weights (e.g. 16 x 1 x 5 x 512)
			#	feat_fc_video = (weights_attn+1) * feat_fc_video

			feat_fc_video = nn.AvgPool2d([num_segments, 1])(feat_fc_video)  # e.g. 16 x 1 x 1 x 512
			feat_fc_video = feat_fc_video.squeeze(1).squeeze(1)  # e.g. 16 x 512

		return feat_fc_video

	def domain_netD_frame(self, feat, beta): #==================netD=================##maybe we need to add more dropout and linear
		feat_fc_domain_frame = GradReverse.apply(feat, beta[2])
		feat_fc_domain_frame1 = self.fc_feature_domain(feat_fc_domain_frame)
		feat_fc_domain_frame2 = self.relu(feat_fc_domain_frame1)
		feat_fc_domain_frame3 = self.dropout_i(feat_fc_domain_frame2)
		feat_fc_domain_frame4 = self.fc_feature_domain2(feat_fc_domain_frame3)
		feat_fc_domain_frame5 = self.relu(feat_fc_domain_frame4)
		feat_fc_domain_frame6 = self.dropout_i(feat_fc_domain_frame5)
		return feat_fc_domain_frame6

	def class_netC_video(self,feat):  #=========================netC==================#
		feat_fc_class_video = self.fc_feature_class_video(feat)
		feat_fc_class_video1 = self.relu(feat_fc_class_video)
		feat_fc_class_video2 = self.dropout_v(feat_fc_class_video1)
		return feat_fc_class_video2

	def domain_classifier_video(self, feat_video, beta):
		feat_fc_domain_video = GradReverse.apply(feat_video, beta[1])
		feat_fc_domain_video = self.fc_feature_domain_video(feat_fc_domain_video)
		feat_fc_domain_video = self.relu(feat_fc_domain_video)
		pred_fc_domain_video = self.fc_classifier_domain_video(feat_fc_domain_video)

		return pred_fc_domain_video

	def domain_classifier_relation(self, feat_relation, beta):
		# 128x4x256 --> (128x4)x2
		pred_fc_domain_relation_video = None
		for i in range(len(self.relation_domain_classifier_all)):
			feat_relation_single = feat_relation[:,i,:].squeeze(1) # 128x1x256 --> 128x256
			feat_fc_domain_relation_single = GradReverse.apply(feat_relation_single, beta[0]) # the same beta for all relations (for now)

			pred_fc_domain_relation_single = self.relation_domain_classifier_all[i](feat_fc_domain_relation_single)
	
			if pred_fc_domain_relation_video is None:
				pred_fc_domain_relation_video = pred_fc_domain_relation_single.view(-1,1,2)
			else:
				pred_fc_domain_relation_video = torch.cat((pred_fc_domain_relation_video, pred_fc_domain_relation_single.view(-1,1,2)), 1)
		
		pred_fc_domain_relation_video = pred_fc_domain_relation_video.view(-1,2)

		return pred_fc_domain_relation_video


	def forward(self, input_source, input_target, beta, mu, is_train, reverse):
		batch_source = input_source.size()[0]
		batch_target = input_target.size()[0]
		num_segments = self.train_segments if is_train else self.val_segments
		# sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
		sample_len = self.new_length
		
		pred_domain_all_source = []
		pred_domain_all_target = []

		# input_data is a list of tensors --> need to do pre-processing
		feat_base_source = input_source.view(-1, input_source.size()[-1]).cuda() # e.g. 64 x 5 x 2048 --> 320 x 2048
		feat_base_target = input_target.view(-1, input_target.size()[-1]).cuda()  # e.g. 64 x 5 x 2048 --> 320 x 2048

		#=== shared layers ===#
		# need to separate BN for source & target ==> otherwise easy to overfit to source data
		if self.add_fc < 1:
			raise ValueError(Back.RED + 'not enough fc layer')

		#==========netB=================#
		feat_fc_source0 = self.fc_feature_shared_source(feat_base_source)
		feat_fc_target0 = self.fc_feature_shared_source(feat_base_target)

		feat_fc_source1 = self.relu(feat_fc_source0)
		feat_fc_target1 = self.relu(feat_fc_target0)
		feat_fc_source = self.dropout_i(feat_fc_source1)
		feat_fc_target = self.dropout_i(feat_fc_target1)

		feat_fc_source = Variable(feat_fc_source, requires_grad=True)
		feat_fc_target = Variable(feat_fc_target, requires_grad=True)
		#feat_fc_source.retain_grad()
		#feat_fc_target.retain_grad()

		# feat_fc = self.dropout_i(feat_fc)
		# if add fc,remember to change the name for each for backward

		#==============netD=======================#
		# === adversarial branch (frame-level) ===#
		fc_domain_sD_source = self.domain_netD_frame(feat_fc_source, beta)
		fc_domain_sD_target = self.domain_netD_frame(feat_fc_target, beta)

		#==============netDL======================##Linear
		pred_fc_domain_frame_source = self.fc_classifier_domain(fc_domain_sD_source)
		pred_fc_domain_frame_target = self.fc_classifier_domain(fc_domain_sD_target)
		logitsDs = pred_fc_domain_frame_source.view((batch_source, num_segments) + pred_fc_domain_frame_source.size()[-1:])
		logitsDt = pred_fc_domain_frame_target.view((batch_target, num_segments) + pred_fc_domain_frame_target.size()[-1:])
		
		#==============netDV======================##Linear+SoftPlus
		varD_source0 = self.variance_domain(fc_domain_sD_source)
		varD_target0 = self.variance_domain(fc_domain_sD_target)
		varD_source = self.softplus(varD_source0)
		varD_target = self.softplus(varD_target0)
		varDs = varD_source.view((batch_source, num_segments,1))
		varDt = varD_target.view((batch_target, num_segments,1))

		#=== backward() compute gradients and new feature ===###only use for training
		if is_train:
			varD_source.backward(torch.ones_like(varD_source),retain_graph=True)
			gradients_source = feat_fc_source.grad #-d VarD/d fi
			pi_source = torch.mul(gradients_source, feat_fc_source) #pi = fi*(-d Var/d fi)
			ai_source0 = torch.add(self.relu(pi_source),torch.mul(self.relu(torch.neg(pi_source)),-1000)) #ai = ReLU(pi)-1000*ReLU(-pi)
			# here we change the dim of our weight to make attention on both time and space
			ai_source1 = ai_source0.view(batch_source, -1) #b_s*(5*512)
			ai_source = self.softmax(ai_source1).view(feat_fc_source.size()) #each picture weight sum is 1

			
			temp_s = torch.sub(torch.ones_like(varD_source),varD_source)
			temp2_s = (temp_s.view(feat_fc_source.size()[0],1)).expand(feat_fc_source.size()[0],feat_fc_source.size()[1])
			#wi_source = torch.mul(temp2_s,self.softmax(ai_source0)) #wi=(1-viD)*Softmax(ai)
			wi_source = torch.mul(temp2_s,ai_source)
			hi_source = torch.mul(feat_fc_source, torch.add(torch.ones_like(wi_source),wi_source)) #hi=fi*(1+wi)

			varD_target.backward(torch.ones_like(varD_target),retain_graph=True)
			gradients_target = feat_fc_target.grad #-d VarD/d fi
			pi_target = torch.mul(gradients_target, feat_fc_target) #pi = fi*(-d Var/d fi)
			ai_target0 = torch.add(self.relu(pi_target),torch.mul(self.relu(torch.neg(pi_target)),-1000)) #ai = ReLU(pi)-1000*ReLU(-pi)
			# here we change the dim of our weight to make attention on both time and space
			ai_target1 = ai_target0.view(batch_target, -1) #b_t*(5*512)
			ai_target = self.softmax(ai_target1).view(feat_fc_target.size()) #each picture weight sum is 1

			temp_t = torch.sub(torch.ones_like(varD_target),varD_target)
			temp2_t = (temp_t.view(feat_fc_target.size()[0],1)).expand(feat_fc_target.size()[0],feat_fc_target.size()[1])
			#wi_target = torch.mul(temp2_t,self.softmax(ai_target0)) #wi=(1-viD)*Softmax(ai)
			wi_target = torch.mul(temp2_t,ai_target)
			hi_target = torch.mul(feat_fc_target, torch.add(torch.ones_like(wi_target),wi_target)) #hi=fi*(1+wi)
		else:
			hi_source = feat_fc_source
			hi_target = feat_fc_target

#=====for attention only in the dimension of space, just use the code below==========#
		# if is_train:
			# varD_source.backward(torch.ones_like(varD_source),retain_graph=True)
			# gradients_source = feat_fc_source.grad #-d VarD/d fi
			# pi_source = torch.mul(gradients_source, feat_fc_source) #pi = fi*(-d Var/d fi)
			# ai_source = torch.add(self.relu(pi_source),torch.mul(self.relu(torch.neg(pi_source)),-1000)) #ai = ReLU(pi)-1000*ReLU(-pi)
			# temp_s = torch.sub(torch.ones_like(varD_source),varD_source)
			# temp2_s = (temp_s.view(feat_fc_source.size()[0],1)).expand(feat_fc_source.size()[0],feat_fc_source.size()[1])
			# wi_source = torch.mul(temp2_s,self.softmax(ai_source)) #wi=(1-viD)*Softmax(ai)
			# hi_source = torch.mul(feat_fc_source, torch.add(torch.ones_like(wi_source),wi_source)) #hi=fi*(1+wi)

			# varD_target.backward(torch.ones_like(varD_target),retain_graph=True)
			# gradients_target = feat_fc_target.grad #-d VarD/d fi
			# pi_target = torch.mul(gradients_target, feat_fc_target) #pi = fi*(-d Var/d fi)
			# ai_target = torch.add(self.relu(pi_target),torch.mul(self.relu(torch.neg(pi_target)),-1000)) #ai = ReLU(pi)-1000*ReLU(-pi)
			# temp_t = torch.sub(torch.ones_like(varD_target),varD_target)
			# temp2_t = (temp_t.view(feat_fc_target.size()[0],1)).expand(feat_fc_target.size()[0],feat_fc_target.size()[1])
			# wi_target = torch.mul(temp2_t,self.softmax(ai_target)) #wi=(1-viD)*Softmax(ai)
			# hi_target = torch.mul(feat_fc_target, torch.add(torch.ones_like(wi_target),wi_target)) #hi=fi*(1+wi)
		# else:
			# hi_source = feat_fc_source
			# hi_target = feat_fc_target



		#=== aggregate the frame-based features to video-based features ###
		if self.frame_aggregation == 'avgpool':
			feat_fc_video_source = self.aggregate_frames(hi_source, num_segments, pred_fc_domain_frame_source)
			feat_fc_video_target = self.aggregate_frames(hi_target, num_segments, pred_fc_domain_frame_target)
		elif self.frame_aggregation == 'rnn':
			feat_fc_video_source = self.aggregate_frames(feat_fc_source, num_segments, pred_fc_domain_frame_source)
			feat_fc_video_target = self.aggregate_frames(feat_fc_target, num_segments, pred_fc_domain_frame_target)

			attn_relation_source = feat_fc_video_source[:,0] # assign random tensors to attention values to avoid runtime error
			attn_relation_target = feat_fc_video_target[:,0] # assign random tensors to attention values to avoid runtime error

		elif 'trn' in self.frame_aggregation:
			feat_fc_video_source = feat_fc_source.view((-1, num_segments) + feat_fc_source.size()[-1:])  # reshape based on the segments (e.g. 640x512 --> 128x5x512)
			feat_fc_video_target = feat_fc_target.view((-1, num_segments) + feat_fc_target.size()[-1:])  # reshape based on the segments (e.g. 640x512 --> 128x5x512)

			feat_fc_video_relation_source = self.TRN(feat_fc_video_source) # 128x5x512 --> 128x5x256 (256-dim. relation feature vectors x 5)
			feat_fc_video_relation_target = self.TRN(feat_fc_video_target)

			# adversarial branch
			pred_fc_domain_video_relation_source = self.domain_classifier_relation(feat_fc_video_relation_source, beta)
			pred_fc_domain_video_relation_target = self.domain_classifier_relation(feat_fc_video_relation_target, beta)

			# transferable attention
			if self.use_attn != 'none': # get the attention weighting
				feat_fc_video_relation_source, attn_relation_source = self.get_attn_feat_relation(feat_fc_video_relation_source, pred_fc_domain_video_relation_source, num_segments)
				feat_fc_video_relation_target, attn_relation_target = self.get_attn_feat_relation(feat_fc_video_relation_target, pred_fc_domain_video_relation_target, num_segments)
			else:
				attn_relation_source = feat_fc_video_relation_source[:,:,0] # assign random tensors to attention values to avoid runtime error
				attn_relation_target = feat_fc_video_relation_target[:,:,0] # assign random tensors to attention values to avoid runtime error

			# sum up relation features (ignore 1-relation)
			feat_fc_video_source = torch.sum(feat_fc_video_relation_source, 1)
			feat_fc_video_target = torch.sum(feat_fc_video_relation_target, 1)
		#print(feat_fc_video_source.size())
		#print(feat_fc_video_target.size())
		#===============netC=========================##Linear+relu+dropout
		#=== source layers (video-level) ===#
		feat_fc_video_source1 = self.dropout_v(feat_fc_video_source)
		feat_fc_video_target1 = self.dropout_v(feat_fc_video_target)

		feat_sC_source = self.class_netC_video(feat_fc_video_source1) #after adpated
		feat_sC_target = self.class_netC_video(feat_fc_video_target1) #after adpated
		adapted_source_feat = feat_sC_source.view(batch_source, -1)
		adapted_target_feat = feat_sC_target.view(batch_target, -1)


		#===============netCL========================##Linear
		pred_fc_video_source = self.fc_classifier_video_source(feat_sC_source)
		pred_fc_video_target = self.fc_classifier_video_source(feat_sC_target)

		#===============netCV========================##Linear+SoftPlus
		varC_source0 = self.variance_class(feat_sC_source)
		varC_target0 = self.variance_class(feat_sC_target)
		varC_source = self.softplus(varC_source0)
		varC_target = self.softplus(varC_target0)

		return pred_fc_video_source, logitsDs, varC_source, varDs, pred_fc_video_target, logitsDt, varC_target, varDt, adapted_source_feat, adapted_target_feat
