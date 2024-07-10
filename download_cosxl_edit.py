from pipelines.cosxl_customized_pipeline import CosStableDiffusionXLInstructPix2PixPipeline
from huggingface_hub import hf_hub_download

edit_file = hf_hub_download(repo_id="stabilityai/cosxl", filename="cosxl_edit.safetensors")

pipe_edit = CosStableDiffusionXLInstructPix2PixPipeline.from_single_file(
    edit_file, num_in_channels=8
)

pipe_edit.to("cuda")
pipe_edit.save_pretrained("preset/CosXL_edit")




