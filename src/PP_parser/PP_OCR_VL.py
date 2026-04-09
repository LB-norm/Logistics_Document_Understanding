from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL()

results = pipeline.predict("data/Lieferschein-Beispiel.png")