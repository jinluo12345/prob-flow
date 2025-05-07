from huggingface_hub import HfApi

hf_api = HfApi(
    endpoint="https://hf-mirror.com",  # 指向你的镜像站
)
local_dir = hf_api.snapshot_download(
    repo_id="pickapic-anonymous/pickapic_v1",
    repo_type="dataset",
    resume_download=True
)
print(f"数据已下载到: {local_dir}")
