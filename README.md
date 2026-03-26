# Attention-Guided Energy Optimization for Label-Aligned Anomaly Generation
This repository implements an attention-guided anomaly synthesis framework for generating image-mask pairs with better label alignment. The method targets the mask label drift problem, where the generated anomalous region does not accurately match the provided ground-truth mask.

Instead of relying on an auxiliary pre-trained segmentation model, our approach directly leverages the diffusion model’s cross-attention maps and introduces an Attention Discrepancy Maximisation (ADM) energy function. During sampling, the method explicitly enlarges the attention discrepancy between target and non-target regions, encouraging the generated anomaly to stay tightly aligned with the GT mask.

By removing the dependency on an extra segmentor, this framework provides a simpler and more robust way to improve image-mask consistency, producing higher-quality synthetic samples for downstream anomaly segmentation tasks. Experiments on multiple datasets show superior generation quality and stronger downstream segmentation performance compared with existing baselines.
