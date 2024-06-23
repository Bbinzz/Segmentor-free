"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor,op_mask

from einops import rearrange
from PIL import Image
import os
import cv2
class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=50,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([
                            unconditional_conditioning[k][i],
                            c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([
                                unconditional_conditioning[k],
                                c[k]])
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            # output,attn = self.model.apply_model(x_in, t_in, c_in)
            # model_uncond, model_t = output.chunk(2)
            model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)

            # classifier free guidance
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)
            # self.visualize_cross_attention(index,attn)

            # my classifier free guidance
            # model_output = model_uncond + unconditional_guidance_scale * (model_uncond - model_t)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
        num_reference_steps = self.ddpm_num_timesteps if use_original_steps else self.ddim_timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), i, device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec
    def visualize_cross_attention(self,index,all_multihead_attn_map, root="/home/wzb/workspace/ControlNet-main/cross_attention_map/self_attnloss[2]/"):
        import matplotlib.pyplot as plt
        import numpy as np
        import cv2
        import os

        for layer_index,multihead_attn_map in enumerate(all_multihead_attn_map):
            save_dir = os.path.join(root,"layer"+str(layer_index),str(index))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            sliced_tensor = np.split(multihead_attn_map, 8, axis=0)
            for i,attention_mask in  enumerate(sliced_tensor):
                # attention_mask = multihead_attn_map[i].cpu().numpy()
                attention_mask = attention_mask.cpu().detach().numpy()  # 对每个批次的所有头求均值
                # print(attention_mask.shape)
                attention_mask = np.mean(attention_mask,axis=0)
                attention_mask = attention_mask.reshape(int(np.sqrt(attention_mask.shape[0])),-1,77)
                attention_mask = attention_mask[:,:,2]

                # mask = cv2.resize(attention_mask,(512,512))
                mask = attention_mask
                min_val = np.min(mask)
                max_val = np.max(mask)
                normed_mask = (mask - min_val) / (max_val - min_val)
                normed_mask = (normed_mask * 255).astype('uint8')
                plt.imshow(normed_mask, alpha=None, interpolation='nearest', cmap="jet")
                # fig, ax = plt.subplots(figsize=(8, 8), dpi=100)  # 设置图形大小为512x512像素
                # binary_map = (average_map >= threshold).astype(float)  # 将平均注意力图二值化

                plt.savefig(f"{save_dir}/binary_attention_map_batch_{i+1}.png")

                plt.close()




class MyDDIMSampler(DDIMSampler):
    def __init__(self,model, schedule="linear",need_mg = False ,**kwargs):
        super().__init__(model, schedule="linear")
        self.mg = need_mg
    @staticmethod
    def _compute_loss():
        pass
    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents
    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, dict):
                assert isinstance(unconditional_conditioning, dict)
                c_in = dict()
                for k in c:
                    if isinstance(c[k], list):
                        c_in[k] = [torch.cat([
                            unconditional_conditioning[k][i],
                            c[k][i]]) for i in range(len(c[k]))]
                    else:
                        c_in[k] = torch.cat([
                                unconditional_conditioning[k],
                                c[k]])
            elif isinstance(c, list):
                c_in = list()
                assert isinstance(unconditional_conditioning, list)
                for i in range(len(c)):
                    c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            # output,attn = self.model.apply_model(x_in, t_in, c_in)
            # model_uncond, model_t = output.chunk(2)
            model_uncond, model_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)

            # classifier free guidance
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)
            # self.visualize_cross_attention(index,attn)
            # my classifier free guidance
            # model_output = model_uncond + unconditional_guidance_scale * (model_uncond - model_t)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    def  p_sample_ddim_mg(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None,mask= None,use_mg=False):
        from ldm.modules.diffusionmodules.util import noise_like
        b, *_, device = *x.shape, x.device

        x.requires_grad = True
        self.model.requires_grad_(True)
        # self.model.decode_first_stage.decoder.requires_grad_(True)
        # mask = torch.squeeze(c["c_concat"][0][:,1,:,:])  # mask 的维度为：[b, 512, 512]
        # mask = torch.unsqueeze(c["c_concat"][0][:,1,:,:],dim=1)  
        # mask = c["c_concat"][0]
        # use_mg = self.if_usegudience(mask)

        if index >= 80:
            repeat = 3
        elif 80 > index >= 40:
            repeat = 1
        else:
            repeat = 1 
        start = 100
        end = 80
        if use_mg== False:
            repeat = 1


        for j in range(repeat):

            x = x.detach().requires_grad_(True)
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                model_output = self.model.apply_model(x, t, c)
            else:
                model_t,_ = self.model.apply_model(x, t, c)
                model_uncond,attnmap_nc = self.model.apply_model(x, t, unconditional_conditioning)
                correction = model_t - model_uncond
                model_output = model_uncond + unconditional_guidance_scale * correction
                # 可视化 attention map
                # self.visualize_cross_attention(index,attnmap_nc,b)
                # self.visualize_self_attention(index,attnmap_nc,b)
                # 保留使用的attention map
                used_attnmap = attnmap_nc[-1]
                if type(used_attnmap) != list:
                    used_attnmap = [used_attnmap]

            if self.model.parameterization == "v":
                e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
            else:
                e_t = model_output

            if score_corrector is not None:
                assert self.model.parameterization == "eps", 'not implemented'
                e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            beta_t = a_t / a_prev
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            # current prediction for x_0
            if self.model.parameterization != "v":
                pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            else:
                pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

            if dynamic_threshold is not None:
                raise NotImplementedError()

            if start > index >= end and use_mg:

                    residual_topk = self.mask_crossattnloss_topk(used_attnmap,mask,b,ratio=1)
                    residual_mean = self.mask_crossattnloss_mean(used_attnmap,mask,b)
                    a = 0.5
                    residual = a*residual_mean +(1-a)*residual_topk


                    # residual = self.mask_crossattnloss_topk(used_attnmap,mask,b)

                    # residual = self.mask_crossattnloss_mean(used_attnmap,mask,b)
                    residual.requires_grad_(True)
                    norm_grad = torch.autograd.grad(outputs=residual, inputs=x,allow_unused=True)[0]
                    rho = (correction * correction).mean().sqrt().item() * unconditional_guidance_scale 
                    rho = rho / (norm_grad * norm_grad).mean().sqrt().item() * 0.08  # 0.03 / 0.08

            c1 = a_prev.sqrt() * (1 - a_t / a_prev) / (1 - a_t)
            c2 = (a_t / a_prev).sqrt() * (1 - a_prev) / (1 - a_t)
            c3 = (1 - a_prev) * (1 - a_t / a_prev) / (1 - a_t)
            c3 = (c3.log() * 0.5).exp()
            x_prev = c1 * pred_x0 + c2 * x + c3 * torch.randn_like(pred_x0)

            if start > index >= end and use_mg:
                x_prev = x_prev - rho * norm_grad.detach()
            
            x = beta_t.sqrt() * x_prev + (1 - beta_t).sqrt() * noise_like(x.shape, device, repeat_noise)

        return x_prev.detach(), pred_x0.detach()
    

    # @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None):
        from tqdm import tqdm
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        # count1 = 0
        # savepath = "/home/wzb/workspace/ControlNet-main/intermediates/stage"
        # if not os.path.exists(savepath):
        #     os.makedirs(savepath)

        
        for i, step in enumerate(iterator):
            
            with torch.enable_grad():
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                
                if mask is not None:
                    assert x0 is not None
                    img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                    img = img_orig * mask + (1. - mask) * img

                if ucg_schedule is not None:
                    assert len(ucg_schedule) == len(time_range)
                    unconditional_guidance_scale = ucg_schedule[i]
                if self.mg:
                    mymask = cond["c_concat"][0]
                    use_mg = self.if_usegudience(mymask,1)
                    outs = self.p_sample_ddim_mg(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        dynamic_threshold=dynamic_threshold,
                                        mask=mymask,
                                        use_mg=use_mg)
                else:
                    outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        dynamic_threshold=dynamic_threshold)
                img, pred_x0 = outs


                # # 可视化中间结果
                # count1 += 1
                # pred_x0_temp = self.model.decode_first_stage(pred_x0)
                # pred_x0_temp = torch.clamp((pred_x0_temp + 1.0) / 2.0, min=0.0, max=1.0)
                # pred_x0_temp = pred_x0_temp.cpu().permute(0, 2, 3, 1).detach().numpy()
                # pred_x0_torch = torch.from_numpy(pred_x0_temp).permute(0, 3, 1, 2)
                # count2 = 0
                # for x_sample in pred_x0_torch:
                #     count2 += 1
                #     x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                #     img_save = Image.fromarray(x_sample.astype(np.uint8))
                #     img_save.save(os.path.join(savepath, "{}_{}.png".format(count1, count2)))

                if callback: callback(i)
                if img_callback: img_callback(pred_x0, i)

                if index % log_every_t == 0 or index == total_steps - 1:
                    intermediates['x_inter'].append(img)
                    intermediates['pred_x0'].append(pred_x0)

        return img, intermediates
    @ staticmethod
    def visualize_cross_attention(index,all_multihead_attn_map,b, root="/home/wzb/workspace/ControlNet-main/SD_cross_attention_map/Controlnet_SDM_tokenindex=1_s=100_e=80_r=3_topk*0.2+mean_nomlization"):
        import matplotlib.pyplot as plt
        import numpy as np
        import cv2
        import os

        for layer_index,multihead_attn_map in enumerate(all_multihead_attn_map):
            save_dir = os.path.join(root,"layer"+str(layer_index),str(index))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            sliced_tensor = np.split(multihead_attn_map, b, axis=0)
            for i,attention_mask in  enumerate(sliced_tensor):
                # attention_mask = multihead_attn_map[i].cpu().numpy()
                attention_mask = attention_mask.cpu().detach().numpy()  # 对每个批次的所有头求均值
                # print(attention_mask.shape)
                attention_mask = np.mean(attention_mask,axis=0)
                attention_mask = attention_mask.reshape(int(np.sqrt(attention_mask.shape[0])),-1,77)
                attention_mask = attention_mask[:,:,1]

                mask = attention_mask
                min_val = np.min(mask)
                max_val = np.max(mask)
                normed_mask = (mask - min_val) / (max_val - min_val)
                # normed_mask = z_score_normalization(attention_mask)
                normed_mask = (normed_mask * 255).astype('uint8')
                plt.imshow(normed_mask, alpha=None, interpolation='nearest', cmap="jet")
                # fig, ax = plt.subplots(figsize=(8, 8), dpi=100)  # 设置图形大小为512x512像素
                # binary_map = (average_map >= threshold).astype(float)  # 将平均注意力图二值化
                counter = 1
                save_path = f"{save_dir}/binary_attention_map_batch_{i+1}_repeat={counter}.png"
                while os.path.exists(save_path):
                    counter += 1
                    save_path = f"{save_dir}/binary_attention_map_batch_{i+1}_repeat={counter}.png"
                plt.savefig(save_path)

                plt.close()
    
    @staticmethod
    def visualize_self_attention(index,multihead_attn_map,batchsize,threshold=0.5, root="/home/wzb/workspace/ControlNet-main/SD_selfattentionmap/selfattnloss"):
        import matplotlib.pyplot as plt
        import numpy as np
        import cv2
        import os
        for layer_index,multihead_attn_map in enumerate(multihead_attn_map):
            save_dir = os.path.join(root,"layer"+str(layer_index),str(index))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            sliced_tensor = np.split(multihead_attn_map, batchsize, axis=0)
            for i,attention_mask in  enumerate(sliced_tensor):
            
                attention_mask = attention_mask.cpu().detach().numpy()  # 对每个批次的所有头求均值
                # print(attention_mask.shape)
                attention_mask = np.mean(attention_mask,axis=0)
                attention_mask = attention_mask.reshape(attention_mask.shape[0],int(np.sqrt(attention_mask.shape[1])),-1)
                attention_mask = np.mean(attention_mask,axis=0)

                # mask = cv2.resize(attention_mask,(512,512))
                mask = attention_mask
                min_val = np.min(mask)
                max_val = np.max(mask)
                normed_mask = (mask - min_val) / (max_val - min_val)
                normed_mask = (normed_mask * 255).astype('uint8')
                plt.imshow(normed_mask, alpha=None, interpolation='nearest', cmap="jet")
                # fig, ax = plt.subplots(figsize=(8, 8), dpi=100)  # 设置图形大小为512x512像素
                # binary_map = (average_map >= threshold).astype(float)  # 将平均注意力图二值化

                plt.savefig(f"{save_dir}/binary_attention_map_batch_{i+1}.png")

                plt.close()
    
    @staticmethod
    def if_usegudience(mask,fiter):
        # 计算等于0的元素数量
        total = mask.numel()
        num_zeros = torch.eq(mask, 1).sum()
        ratio = num_zeros/total
        if ratio <= fiter:
            return True
        else:
            return False
    @staticmethod
    def mask_selfattnloss(self_attnmap,mask,b):
        import torch.nn.functional as F
        # 计算 contorl net 前四层的 attention map 区域之间的差异
        loss = 0
        mask = mask[:,1,:,:]
        # 此时的mask代表boudary
        boundary = op_mask(b,mask).unsqueeze(dim=1)
        size = (64,64)
        for attnmap in self_attnmap:
            sliced_tensor = torch.reshape(attnmap, (b, -1, attnmap.shape[-2], attnmap.shape[-1]))
            mean_multihead = sliced_tensor.mean(dim = 1).squeeze()
            mean_final = mean_multihead.mean(dim = 1).squeeze().reshape(b,64,64)

            layer_mask = F.interpolate(boundary, size=size, mode='bilinear', align_corners=False).squeeze()#[b,size,size]

            # bg_pixels = (layer_mask == 0).sum(dim=[1, 2])  # 在第 1 和第 2 维度上求和
            bg_pixels = (layer_mask==0).sum()
            boundary_pixels = size[0]*size[1] - bg_pixels 

            boundary_region = mean_final * layer_mask
            bg_region = mean_final * (1.- layer_mask)

            boundary_mean = boundary_region.sum()/boundary_pixels
            bg_mean = bg_region.sum()/bg_pixels
            # boundary_mean = ployp_region.sum(dim=[1,2])/boundary_pixels
            # bg_mean = bg_region.sum(dim=[1,2])/bg_pixels
            # taget = torch.max(mean_final,dim=-1)-torch.min(mean_final,dim=-1)

            loss += 1- abs(boundary_mean - bg_mean)
        return loss.mean()
    @staticmethod
    def mask_crossattnloss_mean(attnmap,mask,b):
        import torch.nn.functional as F
        loss = 0
        ### 用mask提高前景和背景的差异
        mask = torch.unsqueeze(mask[:,1,:,:],dim=1) 
        for i in range(len(attnmap)):
            size = (int(np.sqrt(attnmap[i].shape[1])),int(np.sqrt(attnmap[i].shape[1])))
            if b != 1:
                layer_croattn = attnmap[i].unfold(0,b,b).permute(3,0,1,2)
            else:
                layer_croattn = attnmap[i].unsqueeze(0)
            mean_croattn = layer_croattn.mean(dim=1).squeeze(1).permute(0,2,1)
            mean_croattn = mean_croattn.reshape(mean_croattn.shape[:-1]+size)
            # tokenIndex 代表不同文字对应的attention map
            tokenIndex = 1
            compute_croatnn = mean_croattn[:,tokenIndex,:,:].squeeze()
            layer_mask = F.interpolate(mask, size=size, mode='bilinear', align_corners=False).squeeze()#[b,size,size]
            if b != 1:
                bg_pixels = (layer_mask == 0).sum(dim=[1, 2])  # 在第 1 和第 2 维度上求和
                ployp_pixels = size[0]*size[1] - bg_pixels 

                ployp_region = compute_croatnn * layer_mask
                bg_region = compute_croatnn * (1.- layer_mask)

                ploy_mean = ployp_region.sum(dim=[1,2])/ployp_pixels
                bg_mean = bg_region.sum(dim=[1,2])/bg_pixels
            else:
                # 对attention map 进行归一化
                # tensor_min = compute_croatnn.min()
                # tensor_max = compute_croatnn.max()
                # compute_croatnn = (compute_croatnn - tensor_min) / (tensor_max - tensor_min)


                bg_pixels = (layer_mask==0).sum()
                ployp_pixels = size[0]*size[1] - bg_pixels 

                ployp_region = compute_croatnn * layer_mask
                bg_region = compute_croatnn * (1.- layer_mask)
                ploy_mean = ployp_region.sum()/ployp_pixels
                bg_mean = bg_region.sum()/bg_pixels

            loss += 1- abs(ploy_mean - bg_mean)

        return loss.mean()
        ###### 用 ContorlNet 的 cross attention 提取的mask对
        # mask = torch.unsqueeze(mask[:,1,:,:],dim=1)
        # size = (int(np.sqrt(attnmap.shape[1])),int(np.sqrt(attnmap.shape[1])))
        # layer_mask = F.interpolate(mask, size=size, mode='bilinear', align_corners=False).squeeze()
        # attnmap = attnmap.unfold(0,b,b).permute(3,0,1,2) 
        # mean_croattn = attnmap.mean(dim=1).squeeze().permute(0,2,1)
        # mean_croattn = mean_croattn.mean(dim = 1) 
        # mean_croattn = mean_croattn.reshape(mean_croattn.shape[:-1]+size)
        # mean = mean_croattn.mean(dim=0)
        # std = mean_croattn.std(dim=0)

        #     # 归一化
        # normalized_tensor = (mean_croattn - mean) / std
        # threshold = 0.5
        # binary_attn = torch.where(normalized_tensor > threshold, torch.tensor(1.0), torch.tensor(0.0))
        # for i in range(mean_croattn.size(0)):
        #     binary_channel = binary_attn[i, :, :]
                
        #         # 将 PyTorch 张量转换为 NumPy 数组，并调整维度顺序
        #     image_array = binary_channel.cpu().numpy()

        #         # 保存图像
        #     cv2.imwrite("batch_{i}.png", image_array)
    @staticmethod
    def calculate_ratio_of_ones(tensor):
        """
        计算 Tensor 中数值为 1 的元素占比。

        参数:
        tensor : torch.Tensor
            输入的 Tensor。

        返回:
        float
            数值为 1 的元素占比。
        """
        # 计算数值为 1 的元素数量
        ones_count = torch.sum(tensor == 1).item()
        
        # 计算总元素数量
        total_elements = tensor.numel()
        
        # 计算数值为 1 的元素占比
        ratio_of_ones = ones_count / total_elements
        
        return ratio_of_ones
    
    @staticmethod
    def mask_crossattnloss_topk(attnmap,mask,b,ratio=0.2):
        import torch.nn.functional as F
        loss = 0
        ### 用mask提高前景和背景的差异
        mask = torch.unsqueeze(mask[:,1,:,:],dim=1) 
        for i in range(len(attnmap)):
            size = (int(np.sqrt(attnmap[i].shape[1])),int(np.sqrt(attnmap[i].shape[1])))
            if b != 1:
                layer_croattn = attnmap[i].unfold(0,b,b).permute(3,0,1,2)
            else:
                layer_croattn = attnmap[i].unsqueeze(0)
            mean_croattn = layer_croattn.mean(dim=1).squeeze(1).permute(0,2,1)
            mean_croattn = mean_croattn.reshape(mean_croattn.shape[:-1]+size)
            # tokenIndex 代表不同文字对应的attention map
            tokenIndex = 1
            compute_croatnn = mean_croattn[:,tokenIndex,:,:].squeeze()
            layer_mask = F.interpolate(mask, size=size, mode='bilinear', align_corners=False).squeeze()#[b,size,size]
            # 选择目标区域内的topk最低，和背景区域内topk最高
            if b != 1:
                bg_pixels = (layer_mask == 0).sum(dim=[1, 2])  # 在第 1 和第 2 维度上求和
                ployp_pixels = size[0]*size[1] - bg_pixels 

                ployp_region = compute_croatnn * layer_mask
                bg_region = compute_croatnn * (1.- layer_mask)
            else:
                # 对attention map 进行归一化
                # tensor_min = compute_croatnn.min()
                # tensor_max = compute_croatnn.max()
                # compute_croatnn = (compute_croatnn - tensor_min) / (tensor_max - tensor_min)

                bg_pixels = (layer_mask==0).sum()
                ployp_pixels = size[0]*size[1] - bg_pixels 

                ployp_region = compute_croatnn * layer_mask
                ployp_region[ployp_region==0] = 2
                bg_region = compute_croatnn * (1.- layer_mask)
                total_k_number = ployp_pixels if ployp_pixels<bg_pixels else bg_pixels
                
                k_number = int(total_k_number*ratio)

                ployp_topk,_ = torch.topk(-ployp_region.view(-1),k_number)
                ployp_topk_mean = torch.mean(-ployp_topk)

                bg_topk,_ = torch.topk(bg_region.view(-1),k_number)
                bg_topk_mean = torch.mean(bg_topk)

                loss += 1- abs(ployp_topk_mean - bg_topk_mean)


                
        return loss.mean()

