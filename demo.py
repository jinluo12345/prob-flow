from code.pipeline_rf import RectifiedFlowPipelineWithVar, RectifiedFlowPipeline
import torch
from train_prob_flow import UNet2DConditionModelWithVariance
unet_path='/remote-home1/lzjjin/project/InstaFlow/flow-model-finetuned/checkpoint-4000/unet'
unet=UNet2DConditionModelWithVariance.from_pretrained(unet_path,torch_dtype=torch.float32)

pipe = RectifiedFlowPipeline.from_pretrained("XCLIU/2_rectified_flow_from_sd_1_5", 
                                             unet=unet,
                                             torch_dtype=torch.float32) 
### switch to torch.float32 for higher quality

pipe.to("cuda")  ### if GPU is not available, comment this line

prompt = "A hyper-realistic photo of a cute cat."

### 2-rectified flow is a multi-step text-to-image generative model.
### It can generate with extremely few steps, e.g, 2-8
### For guidance scale, the optimal range is [1.0, 2.0], which is smaller than normal Stable Diffusion.
### You may set negative_prompts like normal Stable Diffusion
images = pipe(prompt=prompt, 
              negative_prompt="painting, unreal, twisted", 
              num_inference_steps=25, 
              guidance_scale=1.5).images 
images[0].save("./image.png")