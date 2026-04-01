from typing_extensions import final
import torch
import pytorch_lightning as pl
from cldm.model import create_model, load_state_dict
import numpy as np
from PIL import Image
import os
from Mydataset import MyDataset
# from dreame.dreamedataset import dreameDataset
from torch.utils.data import DataLoader
""" global params """
BATCH_SIZE = 1
CONFIG_FILE_PATH = "/models/myinference.yaml"

CKPT_PATH = "xxx/youself"

RESULT_DIR = "xxx/youself"


learning_rate = 1e-5
logger_freq = 300
sd_locked = True
only_mid_control = False
pl.seed_everything(42,workers=True)  # 用来保证每次生成的结果都是相同的

def get_model(cudaindex):
    model = create_model(CONFIG_FILE_PATH).cpu()
    model.load_state_dict(load_state_dict(CKPT_PATH, location='cuda:{}'.format(cudaindex)),strict = False)
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    model.to("cuda:{}".format(cudaindex))
    model.eval()
    return model
def get_data(batch_size,rotated = False , style = False,angle=0):
    dataset = MyDataset('sample',rotated,style=style)
    # datajson = "/data1/wzb/dataset/polyp/polyp_dataset/TrainDataset/data.json"
    # dataroot = "/data1/wzb/dataset/polyp/polyp_dataset/TrainDataset"
    # dataset = MyDataset(datajson,dataroot,mask_rotated=rotated,angle=angle)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader,dataset


def log_local(save_dir, images, batch_idx,maskbatch,angle=0):
    samples_root = os.path.join(save_dir, "images")
    mask_root = os.path.join(save_dir, "masks")
    tmp = os.path.join(save_dir, "tmp")
    # sample = os.path.join(save_dir,"samples")
    for k in images:
        for idx, image in enumerate(images[k]):
            if k == "samples_cfg_scale_9.00":
                image = (image + 1.0) / 2.0
                image = image.permute(1, 2, 0).numpy()
                image = (image * 255).astype(np.uint8)
                filename = "b-{:06}_idx-{}_{}.png".format(batch_idx, idx,angle)
                path = os.path.join(samples_root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(image).save(path)
            if k == 'control':
                mask = maskbatch.permute(1, 2, 0)[:,:,idx].numpy()
                filename = "b-{:06}_idx-{}_{}.png".format(batch_idx, idx,angle)
                path = os.path.join(mask_root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(mask).convert('1').save(path)
            # if k == "denoise_row":
            #     image = (image + 1.0) / 2.0
            #     image = image.numpy()
            #     image = (image * 255).astype(np.uint8)
            #     filename = "b-{:06}_idx-{}.png".format(batch_idx, idx)
            #     path = os.path.join(tmp, filename)
            #     os.makedirs(os.path.split(path)[0], exist_ok=True)
            #     Image.fromarray(image).save(path)

            # if k == "control":
            #     # mask = masks[idx].numpy().astype(np.uint8)
            #     image = image.permute(1, 2, 0)
            #     image = image.squeeze(-1).numpy()
            #     # mask = (image).astype(np.uint8)
            #     image = (image * 255)
            #     # mask = (255- image.squeeze(-1).numpy()).astype(np.uint8) #255- 实现的是翻转
                
            #     # mask = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8) #变为灰度图
            #     mask = (255 - image[:,:,1]).astype(np.uint8)
            #     filename = "b-{:06}_idx-{}.png".format(batch_idx, idx)
            #     path = os.path.join(mask_root, filename)
            #     os.makedirs(os.path.split(path)[0], exist_ok=True)
            #     Image.fromarray(mask).convert('1').save(path)
                


if __name__ == "__main__":
    cudaindex = 0
    with torch.cuda.device(cudaindex):
        model = get_model(cudaindex)
        isrotated =  True
        result_root = RESULT_DIR
        # finaldir = os.path.join(result_root,"Controlnet_SDM_tokenindex=1_s=100_e=80_r=3_topk*0.2+mean_nomlization") # 不同checkpoint对应生成的不同文件
        # finaldir = os.path.join(result_root,"test")
        # finaldir = os.path.join(result_root,"700_rot") # 不同checkpoint对应生成的不同文件
        finaldir = result_root
        os.makedirs(finaldir, exist_ok=True)
        with torch.no_grad():
            with model.ema_scope():
                    count = 0
                # for angle in [45,30,60]:
                #     dataloader,_ = get_data(BATCH_SIZE,isrotated,style=is_style,angle=angle)
                    dataloader,_ = get_data(BATCH_SIZE,isrotated)
                    for idx, batch in enumerate(dataloader):
                        # if count >= 700:
                        #     break
                        # if idx < 30:
                        #     continue
                        if idx == 8:
                            break
                        print(idx) 
                        import time
                        start_time = time.time()
                        images = model.log_images(
                                batch,
                                N=BATCH_SIZE,
                                ddim_steps = 50,
                                    # plot_diffusion_rows = True
                                    # plot_denoise_rows = True
                                unconditional_guidance_scale=9
                            )
                        end_time = time.time()
                        print("pre_image_time:",end_time-start_time)
                        for k in images:
                            if isinstance(images[k], torch.Tensor):
                                images[k] = images[k].detach().cpu()
                                images[k] = torch.clamp(images[k], -1.0, 1.0)

                        log_local(finaldir, images, idx,maskbatch=batch['mask'],angle=0)
                        count += BATCH_SIZE
                        

