import os
import sys
omp_env = os.environ.get("OMP_NUM_THREADS")
if omp_env is not None:
    try:
        if int(omp_env) <= 0:
            os.environ["OMP_NUM_THREADS"] = "1"
    except ValueError:
        os.environ["OMP_NUM_THREADS"] = "1"
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import yaml
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from core.model import build_model
from torchvision import transforms
from torch.utils.data import DataLoader
from core.utils import load_config, debug
from core.dataset.fusion_datasets import Fusion_Datasets
from core.loss.loss import Generator_Loss, Discriminator_Loss
from fusion_dataset_with_masks import MultiModalDataset_WithMasks


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:   # ← 显式指定编码
        config = yaml.safe_load(f)
    return config
def perturb_mask(mask, cfg):
    if not cfg or not bool(cfg.get('enable', False)):
        return mask
    out = (mask > 0.5).float()
    morph_cfg = cfg.get('morph', {})
    p_morph = float(morph_cfg.get('p', 0.0))
    if p_morph > 0.0 and random.random() < p_morph:
        k = int(morph_cfg.get('kernel_size', 3))
        k = max(3, k)
        if k % 2 == 0:
            k += 1
        if random.random() < float(morph_cfg.get('dilate_prob', 0.5)):
            out = F.max_pool2d(out, kernel_size=k, stride=1, padding=k // 2)
        else:
            out = -F.max_pool2d(-out, kernel_size=k, stride=1, padding=k // 2)
    shift_cfg = cfg.get('shift', {})
    p_shift = float(shift_cfg.get('p', 0.0))
    if p_shift > 0.0 and random.random() < p_shift:
        b, c, h, w = out.shape
        max_shift = max(0, int(shift_cfg.get('max_shift', 0)))
        if max_shift > 0:
            shifted = torch.zeros_like(out)
            for i in range(b):
                dy = random.randint(-max_shift, max_shift)
                dx = random.randint(-max_shift, max_shift)
                src_y0 = max(0, -dy)
                src_y1 = min(h, h - dy) if dy >= 0 else h
                src_x0 = max(0, -dx)
                src_x1 = min(w, w - dx) if dx >= 0 else w
                dst_y0 = max(0, dy)
                dst_x0 = max(0, dx)
                dst_y1 = dst_y0 + (src_y1 - src_y0)
                dst_x1 = dst_x0 + (src_x1 - src_x0)
                if src_y1 > src_y0 and src_x1 > src_x0:
                    shifted[i:i + 1, :, dst_y0:dst_y1, dst_x0:dst_x1] = out[i:i + 1, :, src_y0:src_y1, src_x0:src_x1]
            out = shifted
    return out.clamp(0.0, 1.0)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def perturb_modality_pair(img, mask, cfg):
    """
    对单个模态的图像和掩码施加同一组轻微仿射扰动（平移+旋转），
    以模拟轻微未配准输入。
    """
    if not cfg or not bool(cfg.get('enable', False)):
        return img, mask
    p = float(cfg.get('p', 0.0))
    if p <= 0.0 or random.random() >= p:
        return img, mask

    max_shift = max(0, int(cfg.get('max_shift', 0)))
    max_rotate_deg = float(cfg.get('max_rotate_deg', 0.0))
    if max_shift == 0 and max_rotate_deg == 0.0:
        return img, mask

    out_img = torch.empty_like(img)
    out_mask = torch.empty_like(mask)

    for b in range(img.shape[0]):
        tx = random.randint(-max_shift, max_shift) if max_shift > 0 else 0
        ty = random.randint(-max_shift, max_shift) if max_shift > 0 else 0
        angle = random.uniform(-max_rotate_deg, max_rotate_deg) if max_rotate_deg > 0 else 0.0

        out_img[b] = TF.affine(
            img[b],
            angle=angle,
            translate=[tx, ty],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0
        )
        out_mask[b] = TF.affine(
            mask[b],
            angle=angle,
            translate=[tx, ty],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0.0
        )

    out_mask = (out_mask > 0.5).float()
    return out_img, out_mask


def build_optimizer(opt_name, params, lr):
    if opt_name == 'Adam':
        return optim.Adam(params, lr=lr, betas=(0.5, 0.999))
    return eval('optim.' + opt_name)(params, lr)

def train_Discriminator(epoch, opt, datasets_generator, GAN_Model, Discriminator_Loss, Discriminator_Train_config,
                        input_perturb_cfg=None):

    #epoch: 当前的训练轮次，用于跟踪和显示训练进度。
    #opt: 优化器（optimizer），用于更新判别器的参数。
    #datasets_generator: 训练数据生成器，包含批次数据。
    #GAN_Model: 包含生成器（Generator）和判别器（Discriminator）的 GAN 模型。
    #Discriminator_Loss: 判别器的损失函数，用于衡量生成器输出和真实数据的差异。
    #Discriminator_Train_config: 包含训练时的配置参数，如训练轮次、最小损失等。

    for i in GAN_Model.Generator.parameters():  #获取生成器（Generator）模型的所有参数。
        i.requires_grad = False    #设置生成器的所有参数不参与梯度计算，这样在训练判别器时，生成器的权重不会被更新，只会更新判别器的权重。
    for i in GAN_Model.Discriminator.parameters():   #获取判别器（Discriminator）模型的所有参数。
        i.requires_grad = True    #设置判别器的所有参数参与梯度计算，这样判别器的权重会在训练过程中被更新。

    #初始化训练循环
    num_iter = len(datasets_generator)    #获取数据生成器 datasets_generator 中的迭代次数，表示每个 epoch 中有多少个批次数据。
    train_times_per_epoch = Discriminator_Train_config['train_times_per_epoch']   #从 Discriminator_Train_config 配置中读取每个 epoch 中训练判别器的次数。
    min_loss_per_epoch = Discriminator_Train_config['min_loss_per_epoch']   #从 Discriminator_Train_config 配置中读取最小的损失阈值，如果损失低于此阈值则停止训练。
    train_times = 0   #训练的轮次计数器，初始为 0。
    min_loss = min_loss_per_epoch   #用于记录当前训练的最小损失，初始为 min_loss_per_epoch。

    #开始训练循环
    while train_times < train_times_per_epoch and min_loss >= min_loss_per_epoch:
        #在 train_times 小于设定的 train_times_per_epoch 且 min_loss 大于等于设定的 min_loss_per_epoch 的条件下，继续训练判别器。
        allloss = 0   #用于累积每个批次的损失，最终用来计算该 epoch 的平均损失。
        with tqdm(total=num_iter) as train_Discriminator_bar:  #创建一个进度条 train_Discriminator_bar，用来显示训练进度，num_iter 表示迭代的总次数。
            #遍历数据生成器，训练判别器
            for index, joint_input in enumerate(datasets_generator):  #对 datasets_generator 中的每个数据批次进行迭代，index 为当前批次的索引，data 为当前批次的数据。
                if torch.cuda.is_available():
                    #for i in data:
                        #data = data.cuda()   #检查是否有 GPU 可用（torch.cuda.is_available()），如果有，则将数据转移到 GPU（data[i].cuda()）。
                    joint_input = joint_input.cuda()   #将当前批次的数据 joint_input 移动到 GPU 上。
                inf_img  = joint_input[:, 0:3, :, :]
                inf_mask = joint_input[:, 3:4, :, :]
                vis_img  = joint_input[:, 4:7, :, :]
                vis_mask = joint_input[:, 7:8, :, :]

                target = str((input_perturb_cfg or {}).get('target', 'Inf')).lower()
                if target in ('inf', 'both'):
                    inf_img, inf_mask = perturb_modality_pair(inf_img, inf_mask, input_perturb_cfg)
                if target in ('vis', 'both'):
                    vis_img, vis_mask = perturb_modality_pair(vis_img, vis_mask, input_perturb_cfg)
                # 拼接每个模态的图像和掩码（4通道）
                #vis_input = torch.cat([vis_img, vis_mask], dim=1)  # [B, 4, H, W]
                #inf_input = torch.cat([inf_img, inf_mask], dim=1)  # [B, 4, H, W]
                Discriminatorvis_input = vis_img   # [B, 3, H, W]
                Discriminatorinf_input = inf_img   # [B, 3, H, W]
                Generator_vis_input = torch.cat([vis_img, vis_mask], dim=1)  # shape: (B, 4, H, W)
                Generator_inf_input = torch.cat([inf_img, inf_mask], dim=1)  # shape: (B, 4, H, W)
                Discriminator_data = {
                    'Vis': Discriminatorvis_input,
                    'Inf': Discriminatorinf_input,
                    
                }
                
                Generator_data = {
                    'Vis': Generator_vis_input,
                    'Inf': Generator_inf_input,
                    
                }
                Generator_feats, Discriminator_feats, confidence = GAN_Model(Discriminator_data,Generator_data)
                # GAN_Model(data)：将数据传入 GAN 模型（包含生成器和判别器），并获取输出：
                      # Generator_feats: 生成器的特征（生成的假数据）。
                      # Discriminator_feats: 判别器的特征（判别器对数据的输出）。
                      # confidence: 判别器对于每个输入的信心值（通常是 0 或 1，表示是真实数据还是生成数据）。
                D_loss = Discriminator_Loss(Generator_feats, Discriminator_feats, confidence)
                #通过 Discriminator_Loss 函数，计算生成器和判别器的输出与真实数据之间的损失。
                opt.zero_grad()  #清除优化器中的梯度缓存，以便重新计算梯度。
                D_loss.backward()  #计算反向传播的梯度。
                torch.nn.utils.clip_grad_norm_(GAN_Model.Discriminator.parameters(), 1.0)
                opt.step()   #更新判别器的权重。
                allloss = allloss + D_loss   #累加每个批次的损失。
                train_Discriminator_bar.set_description('\tepoch:%s Train_D iter:%s loss:%.5f' %
                                                        (epoch, index, allloss / num_iter))
                #通过 train_Discriminator_bar.set_description() 更新进度条的描述，显示当前 epoch、训练批次和损失值。
                train_Discriminator_bar.update(1)   #更新进度条，显示已完成的进度。
            min_loss = allloss / num_iter    #计算当前 epoch 的平均损失。
            train_times += 1   #训练的轮次计数器加 1。
            torch.cuda.empty_cache()   #清空 GPU 的缓存，以释放内存。
            del D_loss
    return float(min_loss.detach().cpu().item()) if torch.is_tensor(min_loss) else float(min_loss)


def train_Generator(epoch, opt, datasets_generator, GAN_Model, Generator_Loss, Generator_Train_config,
                    input_perturb_cfg=None):
    for i in GAN_Model.Generator.parameters():
        i.requires_grad = True
    for i in GAN_Model.Discriminator.parameters():
        i.requires_grad = False
    # 初始化训练循环
    num_iter = len(datasets_generator)
    train_times_per_epoch = Generator_Train_config['train_times_per_epoch']
    min_loss_per_epoch = Generator_Train_config['min_loss_per_epoch']
    mask_augment_cfg = Generator_Train_config.get('mask_augment', {})
    train_times = 0
    min_loss = min_loss_per_epoch
    # 开始训练循环
    while train_times < train_times_per_epoch and min_loss >= min_loss_per_epoch:
        # train_times < train_times_per_epoch：当训练轮次（train_times）小于设定的最大训练轮次（train_times_per_epoch）时，继续训练生成器。
        # min_loss >= min_loss_per_epoch：当当前的损失（min_loss）大于等于设定的最小损失阈值（min_loss_per_epoch）时，继续训练。
        allloss = 0  #用来记录整个 epoch 中的累计损失，后面会用来计算每个 epoch 的平均损失。
        with tqdm(total=num_iter) as train_Generator_bar:   #使用 tqdm 库来显示训练进度条，num_iter 是数据集的批次数量。
            for index, joint_input in enumerate(datasets_generator):   #通过 enumerate(datasets_generator) 遍历每个训练数据批次，index 是当前批次的索引，data 是当前批次的数据。
                if torch.cuda.is_available(): 
                      #检查是否有可用的 GPU。
                    joint_input = joint_input.cuda()
                inf_img  = joint_input[:, 0:3, :, :]
                inf_mask = joint_input[:, 3:4, :, :]
                vis_img  = joint_input[:, 4:7, :, :]
                vis_mask = joint_input[:, 7:8, :, :]

                target = str((input_perturb_cfg or {}).get('target', 'Inf')).lower()
                if target in ('inf', 'both'):
                    inf_img, inf_mask = perturb_modality_pair(inf_img, inf_mask, input_perturb_cfg)
                if target in ('vis', 'both'):
                    vis_img, vis_mask = perturb_modality_pair(vis_img, vis_mask, input_perturb_cfg)

                inf_mask = perturb_mask(inf_mask, mask_augment_cfg)
                vis_mask = perturb_mask(vis_mask, mask_augment_cfg)
                # 拼接每个模态的图像和掩码（4通道）
                #vis_input = torch.cat([vis_img, vis_mask], dim=1)  # [B, 4, H, W]
                #inf_input = torch.cat([inf_img, inf_mask], dim=1)  # [B, 4, H, W]
                Discriminatorvis_input = vis_img   # [B, 3, H, W]
                Discriminatorinf_input = inf_img   # [B, 3, H, W]
                Generator_vis_input = torch.cat([vis_img, vis_mask], dim=1)  # shape: (B, 4, H, W)
                Generator_inf_input = torch.cat([inf_img, inf_mask], dim=1)  # shape: (B, 4, H, W)
                Discriminator_data = {
                    'Vis': Discriminatorvis_input,
                    'Inf': Discriminatorinf_input,
                    
                }
                
                Generator_data = {
                    'Vis': Generator_vis_input,
                    'Inf': Generator_inf_input,
                    
                }
                Generator_feats, Discriminator_feats, confidence = GAN_Model(Discriminator_data,Generator_data)
                gen_out = Generator_feats.get('Generator_1')
                if gen_out is not None and index % 20 == 0:
                    gmin = gen_out.min().detach().item()
                    gmax = gen_out.max().detach().item()
                    gmean = gen_out.mean().detach().item()
                    print(f"[Monitor] epoch={epoch} iter={index} Generator_1 min={gmin:.4f} max={gmax:.4f} mean={gmean:.4f}")
                # GAN_Model(data)：将当前批次的数据 data 输入到 GAN 模型中。
                # Generator_feats：生成器生成的特征或假数据。
                # Discriminator_feats：判别器的输出（对输入的判断）。
                # confidence：判别器的信心值，表示该数据是否为真实（或生成的）。
                G_loss = Generator_Loss(Discriminator_data, Generator_feats, Discriminator_feats, confidence)
                # 调用生成器损失函数来计算损失。它考虑输入数据 data，生成器输出 Generator_feats，判别器输出 Discriminator_feats 和判别器的 confidence。
                opt.zero_grad()   # 清空优化器中的梯度缓存，因为 PyTorch 会累积梯度。
                G_loss.backward()   # 计算损失的反向传播，计算梯度。
                torch.nn.utils.clip_grad_norm_(GAN_Model.Generator.parameters(), 1.0)
                opt.step()   # 执行一次优化步长，更新生成器的参数。 
                allloss = allloss + G_loss   # 累积当前批次的损失（G_loss）到总损失（allloss）。
                debug(epoch, Generator_data, Generator_feats, mean=Generator_Train_config['mean'], std=Generator_Train_config['std'])
                # 调用调试函数，输出一些调试信息，如当前训练的 epoch、数据和生成器的特征。这里传递了 mean 和 std，可能用于归一化或调试。
                train_Generator_bar.set_description('\tepoch:%s Train_G iter:%s loss:%.5f' %
                                                    (epoch, index, allloss / num_iter))
                # 更新进度条的描述，显示当前 epoch、训练的批次索引（index）和当前的损失值（allloss / num_iter，即当前 epoch 的平均损失）。
                train_Generator_bar.update(1)   # 每处理完一个批次后，更新进度条。
            min_loss = allloss / num_iter   # 计算当前 epoch 的平均损失（allloss / num_iter），并将其作为当前的最小损失。
            train_times += 1   # 增加训练次数（train_times）。
            torch.cuda.empty_cache()
            del G_loss   # 清空 GPU 的缓存，以释放内存。删除当前批次的损失（G_loss）以节省内存。
    return float(min_loss.detach().cpu().item()) if torch.is_tensor(min_loss) else float(min_loss)


def runner():
    project_name = 'GAN_G1_D2'   #定义了一个字符串变量 project_name，用来指定项目名称，这个名称会用作文件保存路径的一部分。
    #创建存储目录
    try:
        os.makedirs(f'D:/Projects/Pytorch_DDcGAN-master/weights/{project_name}/')
        os.makedirs(f'D:/Projects/Pytorch_DDcGAN-master/weights/{project_name}/Generator/')
        os.makedirs(f'D:/Projects/Pytorch_DDcGAN-master/weights/{project_name}/Discriminator/')
    except:
        pass
    #这部分代码尝试创建保存训练权重的目录。首先创建一个总目录 ./weights/{project_name}/，
    #然后在其下分别创建用于存储生成器和判别器权重的子目录 Generator/ 和 Discriminator/。 
    config = load_config(f'D:/Projects/Pytorch_DDcGAN-master/config/{project_name}.yaml')   #加载配置文件
    #print("[Debug] 当前加载的 config 路径为：", f'D:/Projects/Pytorch_DDcGAN-master/config/{project_name}.yaml')
    #print("[Debug] Discriminator_1 input_dim:", config['Struct']['Discriminator_1']['input_weights_Vis_D']['input_dim'])
    GAN_Model = build_model(config)   #使用配置文件初始化 GAN_Model。
    # build_model 函数创建一个包含生成器（Generator）和判别器（Discriminator）的 GAN 模型。
    


    Datasets_config = config['Dataset']   #数据集相关配置。
    Generator_config = config['Generator']   #生成器的配置。
    Discriminator_config = config['Discriminator']   #判别器的配置。
    Base_Train_config = config['Train']['Base']   #基础训练配置（如 batch_size, epoch）。
    Generator_Train_config = config['Train']['Generator']   #生成器的训练配置（如优化器类型、学习率等）。
    Discriminator_Train_config = config['Train']['Discriminator']   #判别器的训练配置。
    input_perturb_cfg = Base_Train_config.get('input_perturb', {})
    seed = int(Base_Train_config.get('seed', 42))
    set_global_seed(seed)

    #print("Discriminator input dim Vis:", config['Struct']['Discriminator_1']['input_weights_Vis']['input_dim']) 

    mean, std = Datasets_config['mean'], Datasets_config['std']   #mean 和 std 分别是数据集的均值和标准差，用于数据标准化（归一化）。
    input_size = Datasets_config['input_size']   #数据集输入图像的尺寸，用于调整图像的大小。
    sensors = Datasets_config['sensors']    # e.g. ["CT", "MRI"]
    root_dir = Datasets_config['root_dir']   # e.g. "D:/Projects/.../data"
    #定义数据加载器
    train_dataset = MultiModalDataset_WithMasks(
        root_dir     = root_dir,
        sensors      = sensors,
        image_suffix = Datasets_config.get('image_suffix', '.png'),
        mask_suffix  = Datasets_config.get('mask_suffix', '_mask.png'),
        input_size   = (input_size, input_size),
        mean         = tuple(mean),
        std          = tuple(std)
    )
    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(seed)
    train_generator = DataLoader(
        train_dataset,
        batch_size  = Base_Train_config['batch_size'],
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True,
        worker_init_fn=seed_worker,
        generator=data_loader_generator
    )   
    #transform=transforms.Compose([transforms.Resize((input_size, input_size)),
                                                                     #transforms.Grayscale(num_output_channels=3),  # ⭐️ 加这一行  
                                                                     #transforms.ToTensor(),
                                                                     #transforms.Normalize(mean, std)]))
    #train_generator = DataLoader(train_dataloader, batch_size=Base_Train_config['batch_size'], shuffle=False)
    # Fusion_Datasets：这是一个自定义的数据加载类，加载并处理数据。它根据配置中的 root_dir 和 sensors 设置加载数据，使用 transform 来处理图像，包括：
         # Resize：调整图像大小为 input_size。
         # ToTensor：将图像转换为张量（Tensor）。
         # Normalize：使用给定的 mean 和 std 对图像进行标准化。
    # DataLoader：创建一个 train_generator，它是 PyTorch 中的标准数据加载器，用于批量加载数据。batch_size 从配置文件中获取，shuffle=False 表示不打乱数据顺序。
    G_Loss = Generator_Loss(Generator_config)   # 定义生成器的损失函数，传入生成器的配置。
    D_Loss = Discriminator_Loss(Discriminator_config)   # 定义判别器的损失函数，传入判别器的配置。

    if torch.cuda.is_available():
        Generator = GAN_Model.Generator.cuda()
        Discriminator = GAN_Model.Discriminator.cuda()   # 如果有可用的 GPU，使用 .cuda() 将生成器（Generator）和判别器（Discriminator）模型移动到 GPU 上加速训练。
    #初始化优化器
    opt_generator = build_optimizer(Generator_Train_config['opt'], Generator.parameters(), Generator_Train_config['lr'])
    opt_discriminator = build_optimizer(
        Discriminator_Train_config['opt'],
        Discriminator.parameters(),
        Discriminator_Train_config['lr']
    )
    # 根据配置中的优化器类型，使用 eval() 动态选择 PyTorch 中的优化器（如 optim.Adam 或 optim.SGD），并将生成器和判别器的参数传递给它们。
    # Generator_Train_config['lr']：从配置文件中获取生成器的学习率。
    # Discriminator_Train_config['lr']：从配置文件中获取判别器的学习率。

    #训练过程
    best_score = float('inf')
    for epoch in range(1, Base_Train_config['epoch'] + 1):
        
        # exit()
        d_epoch_loss = train_Discriminator(
            epoch, opt_discriminator, train_generator, GAN_Model, D_Loss, Discriminator_Train_config,
            input_perturb_cfg=input_perturb_cfg
        )
        g_epoch_loss = train_Generator(
            epoch, opt_generator, train_generator, GAN_Model, G_Loss, Generator_Train_config,
            input_perturb_cfg=input_perturb_cfg
        )
        epoch_score = d_epoch_loss + g_epoch_loss
        if epoch_score < best_score:
            best_score = epoch_score
            torch.save(Generator, f'D:/Projects/Pytorch_DDcGAN-master/weights/{project_name}/Generator/Generator_best.pth')
            torch.save(Discriminator, f'D:/Projects/Pytorch_DDcGAN-master/weights/{project_name}/Discriminator/Discriminator_best.pth')
            print(
                f"[Best] epoch={epoch} score={epoch_score:.6f} "
                f"(D={d_epoch_loss:.6f}, G={g_epoch_loss:.6f})"
            )
        if epoch % 10 == 0:
            torch.save(Generator, f'D:/Projects/Pytorch_DDcGAN-master/weights/{project_name}/Generator/Generator_{epoch}.pth')
            torch.save(Discriminator, f'D:/Projects/Pytorch_DDcGAN-master/weights/{project_name}/Discriminator/Discriminator_{epoch}.pth')
    #训练循环：对于每一个 epoch，调用 train_Discriminator 和 train_Generator 函数分别训练判别器和生成器。
    #保存模型：每经过 10 个 epoch，使用 torch.save() 将当前的生成器和判别器模型保存到指定的文件路径中，文件名包括当前的 epoch。

if __name__ == '__main__':

    runner()



