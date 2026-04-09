from paddleocr import PPStructureV3
from PP_vis import save_ppstructure_visualizations

pipeline = PPStructureV3(
    lang="de",  # or "en"
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    device="cpu"   
)

results = pipeline.predict("data/Lieferschein-Beispiel.png")

save_ppstructure_visualizations(results, "output/vis")

for res in results:
    res.print()
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")

    data = res.json   