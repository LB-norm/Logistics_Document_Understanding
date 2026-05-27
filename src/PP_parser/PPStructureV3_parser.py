from paddleocr import PPStructureV3, PaddleOCRVL
from PP_vis import save_ppstructure_visualizations

img_path = r"data\datasets\cmr_dachser_20260520\train\images\0ff01432-aefe-45a4-8980-e941151f737d_CMR_page_1.jpg"
# pipeline = PPStructureV3(
#     lang="de",  # or "en"
#     use_doc_orientation_classify=True,
#     use_doc_unwarping=True,
#     device="cpu"   
# )

pipeline = PaddleOCRVL(
    pipeline_version="v1.5",
    device="gpu",
    # cpu_threads=8,
    use_layout_detection=True,   # lighter first test
    merge_layout_blocks=False,
    use_queues=False              # simpler for a single image
)

results = pipeline.predict(
    img_path,
    use_doc_preprocessor=True,
    # max_pixels=512 * 512,
    # max_new_tokens=256
)

save_ppstructure_visualizations(results, "output/vis")

for res in results:
    res.print()
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")

    data = res.json   