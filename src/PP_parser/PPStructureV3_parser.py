from paddleocr import PPStructureV3, PaddleOCRVL
from PP_vis import save_ppstructure_visualizations

img_path = "data/Lieferschein-Beispiel.png"
# pipeline = PPStructureV3(
#     lang="de",  # or "en"
#     use_doc_orientation_classify=True,
#     use_doc_unwarping=True,
#     device="cpu"   
# )

pipeline = PaddleOCRVL(
    device="cpu",
    cpu_threads=8,
    use_layout_detection=True,   # lighter first test
    use_queues=False              # simpler for a single image
)

results = pipeline.predict(
    img_path,
    max_pixels=512 * 512,
    max_new_tokens=256
)

save_ppstructure_visualizations(results, "output/vis")

for res in results:
    res.print()
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")

    data = res.json   