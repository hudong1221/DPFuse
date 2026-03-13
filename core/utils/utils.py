import os
import torch
import random
from torchvision import transforms


def randam_string(num):
	letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
	salt = ''
	for i in range(num):
		salt += random.choice(letters)
	return salt


def debug(epoch, input_dict, output_tensors, mean, std):
	output_names = [i for i in output_tensors]
	batch_size, C, W, H = output_tensors[output_names[0]].shape
	for name in output_names:
		output_tensor = output_tensors[name].data.cpu()
		for i in range(batch_size):
			tensor = output_tensor[i, :, :, :]
			input_vis_tensor = input_dict['Vis'][i,0:3, :, :].data.cpu()
			input_inf_tensor = input_dict['Inf'][i,0:3, :, :].data.cpu()
			img_tensor = torch.cat([input_vis_tensor, input_inf_tensor, tensor], dim=2)
			mean_t = torch.FloatTensor(mean).view(3, 1, 1).expand(img_tensor.shape)
			std_t = torch.FloatTensor(std).view(3, 1, 1).expand(img_tensor.shape)
			img_tensor = img_tensor * std_t + mean_t
			untrans = transforms.Compose([transforms.ToPILImage()])
			img = untrans(img_tensor)
			try:
				os.mkdir(f'D:/Projects/Pytorch_DDcGAN-master/debug/{epoch}/')
			except:
				pass
			img.save(f'D:/Projects/Pytorch_DDcGAN-master/debug/{epoch}/{name}_{randam_string(10)}.jpg')
