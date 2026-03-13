import torch
import torch.nn as nn


class Concat_Layer(nn.Module):   # Concat_Layer：该类继承自 nn.Module，是 PyTorch 中的一个模块，用于定义拼接层。
	def __init__(self, layer_name, config):
		# 初始化方法，接受两个参数
		# layer_name：该层的名称，将用于存储输出的字典键。
		# config：包含拼接层配置的字典。包括拼接的模式（mode）和需要拼接的层列表（layers）。
		super(Concat_Layer, self).__init__()
		self.layer_name = layer_name   # 该层的名称，用于在 feats 字典中存储输出
		self.mode = config['mode']   # 选择拼接模式，可能是 'cat', 'avg' 或 'batch'
		self.layers = config['layers']   # 需要拼接的层名列表

	def forward(self, feats):   # 前向传播方法 forward
		feat = [feats[i] for i in self.layers]
		print(f"Before concat: {[feats[i].shape for i in self.layers]}")  # 打印拼接前的形状
		# 从 feats 字典中提取需要拼接的特征图，self.layers 列表包含了这些特征图所在的层的名称。通过列表推导式，遍历 self.layers 中的每个层名，将对应的特征图（feats[i]）提取出来，放入 feat 列表中。
		if self.mode == 'cat':   # 检查拼接模式是否为 'cat'（即拼接），表示在通道维度上进行拼接。
			feats.update({self.layer_name: torch.cat(feat, dim=1)})
			# 对提取出来的特征图（feat）在通道维度（dim = 1）上进行拼接，得到一个新的特征图。
			# dim=1 表示拼接沿着通道维度（C）进行拼接，因此拼接后的特征图在空间维度（H、W）保持不变，通道数增加。
			# feats.update({self.layer_name: torch.cat(feat, dim=1)})：将拼接后的特征图存入 feats 字典，使用 self.layer_name 作为键。
		elif self.mode == 'avg':   # 检查拼接模式是否为 'avg'，表示将特征图拼接后进行平均池化。
			feats.update({self.layer_name: torch.mean(torch.cat(feat, dim=1), dim=1, keepdim=True)})
			# 首先，使用 torch.cat(feat, dim=1) 将多个特征图在通道维度上拼接。
			# 然后，torch.mean(..., dim=1, keepdim=True) 对拼接后的特征图在通道维度上求平均，即沿通道维度对特征图进行平均池化。
			# keepdim=True 保持输出的维度与输入相同，确保输出是一个具有相同空间尺寸（H、W）的张量。
		elif self.mode == 'batch':   # 检查拼接模式是否为 'batch'，表示将特征图拼接成一个新的 batch。
			feats.update({self.layer_name: torch.cat(feat, dim=0)})
		# 将特征图沿 dim=0（即批次维度）拼接。拼接后的特征图在空间维度和通道维度上保持不变，但批次大小会增加。
		print(f"After concat: {feats[self.layer_name].shape}")  # 打印拼接后的形状
		return feats
		# feats 字典：存储了所有层的特征图，每个键是层的名称，对应的值是该层的特征图。
		# 返回更新后的 feats 字典，其中包含了经过拼接操作后的特征图。这个字典可以继续传递给网络的下一层。



class Conv_Block(nn.Module):   # Conv_Block：这个类继承自 nn.Module，是 PyTorch 中定义的一个模块，表示一个卷积块（Conv Block）。
	def __init__(self, layer_name, config):
		super(Conv_Block, self).__init__()   # 调用父类 nn.Module 的初始化方法，确保类正确初始化
		self.layer_name = layer_name   # 存储该层的名称（layer_name），用于在 feats 字典中存储输出。
		self.reuse_times = config['reuse_times']   # 指定该卷积层是否重复使用。这个参数控制卷积块重复使用的次数（如果需要复用）。
		self.use_residual = config['use_residual']   # 是否使用残差连接。若为 True，则输出将与输入相加。
		self.use_bn = config['use_bn']   # 是否使用批量归一化。若为 True，会在卷积层之后添加批量归一化层。
		self.use_activation = config['use_activation']   # 是否使用激活函数，config 中的 use_activation 定义了激活函数的类型（如 ReLU、Tanh 等）。
		self.parameters = config['parameters']   # 存储卷积层的参数配置，如输入输出通道数、卷积核大小等。
		self.out_channels = self.parameters['out_channels']   # 从 parameters 中提取输出通道数。
		print(f"Initializing Conv2d with parameters: {self.parameters}")
		self.conv = nn.Conv2d(**self.parameters)
		# 创建一个二维卷积层，使用 config['parameters'] 中的参数来定义卷积层。
		# **self.parameters 通过解包将字典中的参数传递给 nn.Conv2d，如输入输出通道数、卷积核大小等。
		if self.use_bn:   # 如果配置中指定使用批量归一化（use_bn=True），则添加一个批量归一化层。
			self.bn = nn.BatchNorm2d(self.out_channels)   # 创建一个二维批量归一化层，self.out_channels 是卷积层的输出通道数。
		if self.use_activation != 'None':   # 如果配置中指定了使用激活函数（use_activation 不是 None），则创建激活层。
			self.activation = eval('nn.' + self.use_activation)()   # 动态创建激活函数层。
			# 例如，若 use_activation='ReLU'，则通过 eval('nn.ReLU') 动态创建一个 ReLU 激活函数层。eval() 可以执行字符串形式的代码，这里用来根据字符串名称创建相应的激活函数。

	def forward(self, feats):   # 前向传播方法，输入是 feats 字典，表示网络中各层的输出特征。
		index = [i for i in feats][-1]
		x = feats[[i for i in feats][-1]]  # 从 feats 字典中获取最后一个输入特征（假设这是输入数据，通常是上一个层的输出），用于卷积操作。 
		feat = self.conv(x)   # 通过卷积层对输入 x 进行卷积运算，得到卷积输出 feat。

		if self.use_bn:   # 如果配置中指定使用批量归一化，则将卷积输出 feat 传递给批量归一化层 self.bn(feat)。
			feat = self.bn(feat)
		if self.use_activation != None and self.use_residual:
			# 如果配置中同时指定了激活函数和残差连接，则：
			# 将卷积输出 feat 和输入 x 相加（残差连接）。
			# 对相加后的结果应用激活函数。
			feat = self.activation(feat + x)
		elif self.use_activation == None and self.use_residual:
			# 如果没有激活函数但启用了残差连接，则直接将卷积输出 feat 与输入 x 相加，形成残差连接。
			feat = feat + x
		elif self.use_activation == None and not self.use_residual:
			# 如果没有激活函数且没有使用残差连接，则直接使用卷积输出 feat。
			feat = feat
		feats.update({self.layer_name: feat})   # 将当前层的输出 feat 存储到 feats 字典中，使用 self.layer_name 作为键。

		return feats   # 返回更新后的 feats 字典，包含当前层的输出和网络中其他层的输出。


# 定义多头注意力模块
class haloAttentionBlock(nn.Module):
	"""示例的多头注意力层，用于在 (B, C, H, W) 的特征图上做多头注意力"""
	def __init__(self, layer_name, config):
		super().__init__()
		self.layer_name = layer_name   # 在 feats 字典里存取时用到的键
		self.embed_dim = config['embed_dim']   # 每个 token 的特征维度，等于通道数 C
		self.num_heads = config['num_heads']   # 多头数量
		self.patch_size = config.get('patch_size')  # 块的大小
		self.halo = config.get('halo', 8)  # overlap即重叠区域的大小

		# PyTorch内置的多头注意力
		self.mha = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)   # self.mha 会计算softmax

		# 线性映射 + 层归一化，用于残差连接
		self.fc = nn.Linear(self.embed_dim, self.embed_dim)   # 投影层
		self.norm = nn.LayerNorm(self.embed_dim)   # 层归一化

		# 获取输入层的名称，这里假设从 config['layers'] 获取
        # 如果 config 中的 layers 为 ['layer_1']，则 input_key = 'layer_1'
		self.input_key = config['layers'][0]   # 输入层的名称
        

	def forward(self, feats):
		# feats: 一个字典，保存各层特征。其中 feats[self.layer_name] 是一个 4D Tensor (B, C, H, W)，
		x = feats[self.input_key]   # 形状 (B, C, H, W)
		B, C, H, W = x.shape
		# 如果 embed_dim != C，需要你自己处理 shape 对齐（此处假设 embed_dim == C）
		# 把 (B, C, H, W) flatten 到 (seq_len, batch_size, embed_dim)
		# seq_len = H*W, batch_size = B, embed_dim = C
		# 把空间维度摊平成序列

		p = self.patch_size   # 每个块的大小，必须能整除H和W
		halo = self.halo

		assert H % p == 0 and W % p == 0, "H, W must be multiples of patch_size"
		# 计算块的数量
		nH = H // p
		nW = W // p
		# 创建输出tensor，保持和输入的相同形状
		out = x.new_zeros(x.shape)    # 最终输出 (B, C, H, W)

		for i in range(nH):
			for j in range(nW):
				# 计算当前 patch 的坐标
				# 注意：这里的 patch 是指在原始特征图上划分的区域
				# 例如，patch_size=64 时，(i, j)=(0, 0) 对应的区域是 (0, 0) 到 (64, 64)
				# 而 (i, j)=(1, 1) 对应的区域是 (64, 64) 到 (128, 128)
				row0 = i * p
				col0 = j * p
				
				# 扩展后的patch (中心: p*p, plus halo overlap)
				rowA = max(0, row0 - halo)
				colA = max(0, col0 - halo)
				rowB = min(H, row0 + p + halo)
				colB = min(W, col0 + p + halo)
				# 提取当前 patch 的特征
				patch = x[:, :, rowA:rowB, colA:colB]

				# flatten
				B2, C2, H2, W2 = patch.shape
				patch_reshape = patch.reshape(B2, C2, -1).permute(2, 0, 1) # (H2*W2, B, C2)

				# 做局部注意力
				attn_out, _ = self.mha(patch_reshape, patch_reshape, patch_reshape)
				attn_out = self.fc(attn_out)
				attn_out = self.norm(attn_out + patch_reshape)   # 残差链接

				# 将输出reshape 回原来 (B, C, H2, W2)
				attn_out = attn_out.permute(1, 2, 0).view(B2, C2, H2, W2)
				# 把 attn_out 的中心 p*p 区域放回 out
				# center region 在 patch_reshape 里坐标:
				# row0-rowA : row0-rowA+p -> (p, p)
				# col0-colA : col0-colA+p
				c_row0 = row0 - rowA
				c_col0 = col0 - colA
				# 将计算后的块放回原来的位置
				out[:, :, row0:row0+p, col0:col0+p] = attn_out[:, :, c_row0:c_row0+p, c_col0:c_col0+p]

		feats[self.layer_name] = out   # 更新feats字典，保存输出
		return feats


class input_weights_Layer(nn.Module):
	"""给指定的输入层乘以可学习的权重"""

	def __init__(self, layer_name, config):
		super(input_weights_Layer, self).__init__()
		import traceback
		traceback.print_stack()
		self.layer_name = layer_name
		# config 里要包含哪些输入需要加权（例如 ['Vis'] 或 ['Inf']）
		self.layers = config['layers']
		# 输入的通道数
		print(f"[Debug] layer_name={layer_name}")
		if layer_name == 'input_weights_Vis_D' or layer_name == 'input_weights_Gen_D' or layer_name == 'input_weights_Inf_D':
			# 这里假设输入是 RGB 图像，通道数为 3
			self.input_dim = 3
		else:
			self.input_dim = config['input_dim']
		#self.input_dim = config['input_dim']
		# 可学习权重，初始全 1，形状 (input_dim,)
		print(f"[input_weights_Layer] {self.layer_name} 最终 input_dim={self.input_dim}")
		self.weight = nn.Parameter(torch.ones(self.input_dim))
		

		

	def forward(self, feats):
		# feats: 一个 dict, 里面有 'Vis', 'Inf' 等键。
		# self.layers: 要对这些键对应的张量进行加权
		for key in self.layers:
			x = feats[key]  # 形状 (B, input_dim, H, W)
			# 将 weight reshape 后与 x 相乘
			w = self.weight.view(1, -1, 1, 1)
			x = x * w
			feats.update({key: x})  # 更新回 feats
		return feats


class Model(nn.Module):
	"""docstring for Model"""

	def __init__(self, config):   # 构造函数，接收一个 config 字典作为参数。这个字典通常包含了网络层的配置。
		super(Model, self).__init__()
		# 记录结构信息
		self.model_names = {i: {} for i in config}   # 一个字典，包含了网络模型结构的信息。它的键是模型的名称，值是一个空字典，稍后用于存储各层的信息。
		self.models = nn.ModuleDict({})   # 一个 nn.ModuleDict，它是一个字典，键是模型的名称，值是该模型的实际层模块。
		# 注意：你可能需要在每个 model_name 里各自构造一个新的 model_dict
		# 此处简化写法，直接在循环外声明
		model_dict = nn.ModuleDict({})   # 用于存储各层模块的字典。
		# 遍历所有 'model_name' (例如 'Generator', 'Discriminator' ...)
		for model_name in self.model_names:
			self.layer_names = {i: {} for i in config[model_name]}   # 为当前模型的每一层创建一个空字典。
			# 遍历该 model_name 下的每个 layer
			for layer_name in self.layer_names:   # 遍历当前模型中的每一层。
				layer_type = config[model_name][layer_name]['type']   # 获取当前层的类型（例如 conv，concat，multihead_attn 等）。
				layer_config = config[model_name][layer_name]   # 获取当前层的配置。
				# 根据 layer_type 不同，动态创建相应模块
				if layer_type == 'conv':   # 如果层类型是卷积层，则根据配置创建 Conv_Block。如果 same_weight 为 True，则创建具有相同权重的多个块；如果为 False，则创建多个块并赋予不同的权重。
					if layer_config['same_weight']:
						block = Conv_Block(layer_name, layer_config)
						layer = {'block_' + str(i): block for i in range(layer_config['reuse_times'])}
					else:
						layer = {'block_' + str(i): Conv_Block(layer_name, layer_config)
						         for i in range(layer_config['reuse_times'])}
				elif layer_type == 'concat':
					block = Concat_Layer(layer_name, layer_config)
					layer = {layer_name: block}
				elif layer_type == 'multihead_attn':
					# 处理多头注意力层
					block = haloAttentionBlock(layer_name, layer_config)
					layer = {layer_name: block}
				elif layer_type == 'input_weights':
					block = input_weights_Layer(layer_name, layer_config)
					layer = {layer_name: block}
				#else:
					#raise NotImplementedError(f"Unknown layer type: {layer_type}")

				self.layer_names.update({layer_name: [i for i in layer]})   # 将当前层的名称和该层的块存入 self.layer_names 字典中。
				model_dict.update({layer_name: nn.ModuleDict(layer)})   # 将当前层的模块（例如 Conv_Block，Concat_Layer 等）存入 model_dict 字典。
			self.model_names.update({model_name: self.layer_names})   # 将当前模型的层名称信息存入 self.model_names 中。
			self.models.update({model_name: nn.ModuleDict(model_dict)})   # 将当前模型的层模块存入 self.models 中，供后续使用。

	def forward(self, feats):   # 前向传播，输入为特征字典 feats。
		for model_name in self.model_names:   # 遍历所有的模型、层和块，依次调用每个块的前向传播方法，更新 feats 字典。
			for layer_name in self.model_names[model_name]:
				for block_name in self.model_names[model_name][layer_name]:
					feats = self.models[model_name][layer_name][block_name](feats)
					#print(f"After {block_name}: {feats[self.layer_name].shape}")
		return feats



