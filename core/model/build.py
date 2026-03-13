import torch
import torch.nn as nn
from core.model.base import Model


class build_model(nn.Module):
	def __init__(self, config):
		super(build_model, self).__init__()
		Generator_config = config['Generator']   # 生成器名字列表，如["G1","G2"...]
		self.Generator_name = Generator_config['Generator_Name']
		# 每个生成器对应需要从输入中取哪些数据集
		# 例如: {"G1": ["img"], "G2": ["text"]} 等
		self.Generator_input = {G: input for G, input in zip(self.Generator_name, Generator_config['Input_Datasets'])}
		# self.Generator 是一个 ModuleDict，其中包含了多个生成器子模型
		# 每个 key 即生成器的名字 i，每个 value 为 Model(...) 实例
		self.Generator = nn.ModuleDict({
			i: Model({i: config['Struct'][i]}) for i in self.Generator_name})

		# 初始化 Discriminators
		Discriminator_config = config['Discriminator']
		self.Discriminator_name = Discriminator_config['Discriminator_Name']
		self.Discriminator_input = {D: input for D, input in zip(self.Discriminator_name, Discriminator_config['Input_Datasets'])}
		self.Discriminator = nn.ModuleDict({i: Model({i: config['Struct'][i]}) for i in self.Discriminator_name})

	def forward(self, Discriminator_data,Generator_data):
		# 1) 依次调用各生成器
		Generator_feats = {}    # 一个孔字典，用于储存每一个生成器的特征
		for G in self.Generator:
			# 复制一份 inputs 字典，后面只保留这个生成器需要的键
			Generator_inputs = Generator_data.copy()
			# 根据 self.Generator_input[G] 取出所需的子集
			Generator_inputs = {i: Generator_inputs[i] for i in self.Generator_input[G]}
			# 调用某个生成器子模型，得到其输出（往往是个 dict）
			Generator_feat = self.Generator[G](Generator_inputs)
			# 这里取字典中最后一个键对应的值，作为最终要用的特征
			# [i for i in Generator_feat] 会生成所有键的列表，[-1] 取最后一个
			#if 'multihead_attn_1' in Generator_feat:
				#Generator_feats['multihead_attn_1'] = Generator_feat['multihead_attn_1']
			Generator_feat = Generator_feat[[i for i in Generator_feat][-1]]
			# 将生成器得到的最终特征放入 Generator_feats 字典
			Generator_feats.update({G: Generator_feat})

		# 2) 依次调用各判别器
		Discriminator_feats = {}   # 用于储存每一个判别器的特征
		confidence = {}   # 用于储存每个判别器的置信度
		for D in self.Discriminator:
			Discriminator_inputs = Discriminator_data.copy()   #复制输入字典inputs，用于为每个判别器准备输入数据
			# Discriminator_inputs.update({'Generator': Generator_feats['Generator']})
			Discriminator_inputs.update(Generator_feats)   # 将生成器的输出添加到判别器的输入中
			Discriminator_inputs = {i: Discriminator_inputs[i] for i in self.Discriminator_input[D]}   # 根据 self.Discriminator_input[D] 取出所需的子集
			# 说明判别器的输入中可能会整合生成器的输出
			# 这里根据是否是生成器产生的数据还是原数据，构造 0/1 标记并拼接
			# 如果 i 在 self.Generator 里（表示 i 是生成器名），则为 0；否则为 1
			confidence.update({D: torch.cat(
				[torch.zeros(Discriminator_inputs[i].shape[0]).to(Discriminator_inputs[i].device)
				 if i in self.Generator else
				 torch.ones(Discriminator_inputs[i].shape[0]).to(Discriminator_inputs[i].device)
				 for i in Discriminator_inputs], dim=0)})

			# 正式调用该判别器子模型
			Discriminator_feat = self.Discriminator[D](Discriminator_inputs)
			# if 'multihead_attn_1' in Discriminator_feat:
				# Generator_feats['multihead_attn_1'] = Discriminator_feat['multihead_attn_1']
			# 同样取出字典里最后一个键对应的结果
			Discriminator_feat = Discriminator_feat[[i for i in Discriminator_feat][-1]]
			# squeeze() 去掉多余维度 (比如 [B, 1] -> [B], 具体看判别器输出形状)
			Discriminator_feats.update({D: Discriminator_feat.squeeze()})

		return Generator_feats, Discriminator_feats, confidence
