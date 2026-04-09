# ruff: noqa: E702

from PIL import Image

from models.earthmind import EarthMind
from models.qwen3_vl_8b import Qwen3VL8B
from models.qwen_image_edit import QwenImageEditGenerator
from models.remotesam import RemoteSAM
from models.sam3 import SAM3Segmenter
from pipelines.earthmind import EarthMindPipeline
from pipelines.majority_voting import MajorityVotingPipeline
from pipelines.qwen_sgr import QwenSGRPipeline
from pipelines.remotesam import RemoteSAMPipeline
from pipelines.sgr import SGRPipeline
from pipelines.cgr import CGRPipeline
from pipelines.sam3 import SAM3Pipeline
from prompts import EXTRACT_GROUNDING_INFO_PROMPT, QUERY_CLASSIFIER_PROMPT

# fmt: off
# Models
earthmind = EarthMind(device="cuda:0"); earthmind.load()
earthmind_ft = EarthMind(load_finetuned=True, device="cuda:0"); earthmind_ft.load()
# falcon = Falcon(); falcon.load()
sam3 = SAM3Segmenter(device="cuda:0"); sam3.load()
remotesam = RemoteSAM(device="cuda:0"); remotesam.load()
qwen3_vl_8b = Qwen3VL8B(device="cuda:0"); qwen3_vl_8b.load()
qwen_image_edit = QwenImageEditGenerator(device="cuda:1"); qwen_image_edit.load()

# Pipelines
earthmind_pipeline = EarthMindPipeline(); earthmind_pipeline.model = earthmind
# falcon_pipeline = FalconPipeline(); falcon_pipeline.model = falcon
sam3_pipeline = SAM3Pipeline(); sam3_pipeline.model = sam3
remotesam_pipeline = RemoteSAMPipeline(); remotesam_pipeline.model = remotesam
sgr_pipeline = SGRPipeline(); sgr_pipeline.qwen = qwen3_vl_8b; sgr_pipeline.detector = remotesam; sgr_pipeline.sam3 = sam3
cgr_pipeline = CGRPipeline(); cgr_pipeline.qwen = qwen3_vl_8b; cgr_pipeline.detector = remotesam; cgr_pipeline.sam3 = sam3
qwen_sgr_pipeline = QwenSGRPipeline(); qwen_sgr_pipeline.qwen_image_edit = qwen_image_edit; qwen_sgr_pipeline.sam3 = sam3; qwen_sgr_pipeline.remotesam = remotesam

pipeline = MajorityVotingPipeline(
    pipelines=[
        earthmind_pipeline,
        # falcon_pipeline,
        sam3_pipeline,
        remotesam_pipeline,
        sgr_pipeline,
        cgr_pipeline,
        qwen_sgr_pipeline,
    ],
)
pipeline.load_models()
# fmt: on


def generate_caption(image: Image.Image) -> str:
    # NOTE: Don't add <image> tag here, it is automatically added by `earthmind.answer`
    return earthmind.answer(image, "Describe this image in 50 words.")


def answer_boolean_question(image: Image.Image, question: str) -> bool:
    # NOTE: Don't add <image> tag here, it is automatically added by `earthmind.answer`
    answer = earthmind_ft.answer(image, f"Answer with only 'Yes' or 'No': {question}")
    if (
        "true" in answer.lower()
        or "false" in answer.lower()
        or "yes" in answer.lower()
        or "no" in answer.lower()
    ):
        return "true" in answer.lower() or "yes" in answer.lower()
    else:
        print(f"Warning: Model generated a non-boolean answer for {question}: {answer}")
        return False


def answer_numerical_question(image: Image.Image, question: str) -> int:
    # NOTE: Don't add <image> tag here, it is automatically added by `earthmind.answer`
    answer = earthmind_ft.answer(image, f"Numeric question: {question}")
    answer = answer.strip().split("\n")[-1].strip()
    if answer.isnumeric():
        return int(answer)
    else:
        print(f"Warning: Model generated a non-numeric answer for {question}: {answer}")
        return 1


def answer_text_question(image: Image.Image, question: str) -> str:
    # NOTE: Don't add <image> tag here, it is automatically added by `earthmind.answer`
    return earthmind_ft.answer(image, f"Give straight-forward answer: {question}")


def generate_bboxes(image: Image.Image, question: str) -> list[list[float]]:
    updated_question = qwen3_vl_8b.generate_text_only(
        EXTRACT_GROUNDING_INFO_PROMPT.format(question=question)
    )
    w, h = image.size
    return [
        [
            bbox["oriented_bbox"][0] / w,
            bbox["oriented_bbox"][1] / h,
            bbox["oriented_bbox"][2] / w,
            bbox["oriented_bbox"][3] / h,
            bbox["oriented_bbox"][4] / w,
            bbox["oriented_bbox"][5] / h,
            bbox["oriented_bbox"][6] / w,
            bbox["oriented_bbox"][7] / h,
        ]
        for bbox in pipeline.process_image(image, updated_question)
    ]


def classify_prompt(image: Image.Image, prompt: str) -> int:
    modified_prompt = QUERY_CLASSIFIER_PROMPT.format(question=prompt)
    response = qwen3_vl_8b.generate_text_only(modified_prompt)
    if "caption" in response.lower():
        return 0
    elif "grounding" in response.lower():
        return 1
    elif "binary" in response.lower():
        return 2
    elif "semantic" in response.lower():
        return 3
    elif "numeric" in response.lower():
        return 4
    else:
        return 0
