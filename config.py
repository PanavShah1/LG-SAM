import os

prod_var = os.environ.get("PYTHON_ENV", "dev").lower()
is_production = "prod" in prod_var

if is_production:
    # Local checkpoints for production
    EARTHMIND_CHECKPOINT = "./checkpoints/EarthMind-4B"
    EARTHMIND_FT_CHECKPOINT = "./checkpoints/EarthMind-4B-ft"
    FALCON_CHECKPOINT = "./checkpoints/Falcon-Single-Instruction-Large"
    QWEN3_VL_8B_CHECKPOINT = "./checkpoints/Qwen3-VL-8B-Instruct"
    QWEN_IMAGE_EDIT_CHECKPOINT = "./checkpoints/Qwen-Image-Edit-2509"
    REMOTE_SAM_CHECKPOINT = "./checkpoints/RemoteSAMv1.pth"
    SAM3_CHECKPOINT = "./checkpoints/sam3/sam3.pt"
    BERT_CHECKPOINT = "./checkpoints/bert-base-uncased"
else:
    # Hugging Face checkpoints for development
    EARTHMIND_CHECKPOINT = "./checkpoints/EarthMind-4B"
    EARTHMIND_FT_CHECKPOINT = "./checkpoints/EarthMind-4B-ft"
    FALCON_CHECKPOINT = "./checkpoints/Falcon-Single-Instruction-Large"
    QWEN3_VL_8B_CHECKPOINT = "Qwen/Qwen3-VL-8B-Instruct"
    QWEN_IMAGE_EDIT_CHECKPOINT = "Qwen/Qwen-Image-Edit-2509"
    REMOTE_SAM_CHECKPOINT = "./checkpoints/RemoteSAMv1.pth"
    SAM3_CHECKPOINT = None
    BERT_CHECKPOINT = "bert-base-uncased"
