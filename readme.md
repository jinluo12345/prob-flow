# The implemantation of Flow with probability

## Quick Start
### Download Data
This will automically download pickapick-v1 data. You can also change the data to download.
```bash
python download.py 
```

### Training
```bash
accelerate launch --num_processes 8  --num_machines 1  --multi-gpu --mixed_precision="fp16"  train_prob_flow_pick.py --config config/pickapick-base.yaml 
```
you can change the model or data to be used in the config.

### Inference
Here,unet path is the newly trained model. The pipeline is the most importtant as it generates images. RectifiedFlowPipelineWithVar samples v_pred using distribution and uses var predicted. While RectifiedFlowPipeline only uses mu predicted as v pred.
You can refer to /code/pipeline_rf.py for further detials.
```python
from code.pipeline_rf import RectifiedFlowPipelineWithVar, RectifiedFlowPipeline
import torch
from train_scripts.train_prob_flow import UNet2DConditionModelWithVariance
unet_path='/checkpoint-4000/unet'
unet=UNet2DConditionModelWithVariance.from_pretrained(unet_path,torch_dtype=torch.float32)

pipe = RectifiedFlowPipeline.from_pretrained("XCLIU/2_rectified_flow_from_sd_1_5", 
                                             unet=unet,
                                             torch_dtype=torch.float32) 
### switch to torch.float32 for higher quality

pipe.to("cuda")  ### if GPU is not available, comment this line

prompt = "A hyper-realistic photo of a cute cat."

### For guidance scale, the optimal range is [1.0, 2.0], which is smaller than normal Stable Diffusion.
### You may set negative_prompts like normal Stable Diffusion
images = pipe(prompt=prompt, 
              negative_prompt="painting, unreal, twisted", 
              num_inference_steps=25, 
              guidance_scale=1.5).images 
images[0].save("./image.png")
```