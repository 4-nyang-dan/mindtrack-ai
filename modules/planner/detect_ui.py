import os
import time
import numpy as np
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import easyocr
from joblib import Parallel, delayed


class UIDetector:
    def __init__(self, gpu=False, iou_threshold=0.6, conf_threshold=0.15):
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

        model_path = hf_hub_download(
            "microsoft/OmniParser-v2.0",
            "icon_detect/model.pt",
            repo_type="model"
        )
        self.detector = YOLO(model_path)

        self.ocr = easyocr.Reader(['ko', 'en'], recog_network='korean_g2', gpu=gpu)

    def _merge_boxes(self, boxes):
        merged = []
        for box in boxes:
            x1, y1, x2, y2, conf = box
            added = False
            for i, m in enumerate(merged):
                mx1, my1, mx2, my2, _ = m
                inter_x1 = max(x1, mx1)
                inter_y1 = max(y1, my1)
                inter_x2 = min(x2, mx2)
                inter_y2 = min(y2, my2)

                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - my1)
                    iou = inter_area / min((x2-x1)*(y2-y1), (mx2-mx1)*(my2-my1))
                    if iou > self.iou_threshold:
                        merged[i] = [
                            min(x1, mx1), min(y1, my1),
                            max(x2, mx2), max(y2, my2),
                            conf
                        ]
                        added = True
                        break
            if not added:
                merged.append(box)
        return merged

    def _ocr_single_crop(self, crop):
        w, h = crop.size
        if min(w, h) < 48:
            scale = max(2, int(96/min(w, h)))
            crop = crop.resize((w * scale, h * scale))

        res = self.ocr.readtext(np.array(crop))
        return " ".join([t[1] for t in res]).strip() if res else None

    def extract(self, image_path: str):
        total_start = time.time()
        im = Image.open(image_path).convert("RGB")

        # === YOLO ===
        yolo_start = time.time()
        results = self.detector(image_path, conf=self.conf_threshold)
        yolo_time = time.time() - yolo_start

        boxes = []
        for r in results:
            for b, c in zip(r.boxes.xyxy, r.boxes.conf):
                boxes.append([*map(int, b.tolist()), float(c)])
        boxes = self._merge_boxes(boxes)

        # === OCR SKIP RULES ===
        crops_for_ocr = []
        skip_indices = {}
        indices_for_ocr = []

        for idx, (x1, y1, x2, y2, _) in enumerate(boxes):
            w, h = (x2-x1), (y2-y1)

            # 작은 정사각형 / 아이콘 → OCR Skip
            if (w < 32 and h < 32) or abs(w - h) < 10:
                skip_indices[idx] = None
                continue

            crops_for_ocr.append(im.crop((x1, y1, x2, y2)))
            indices_for_ocr.append(idx)

        # === 병렬 OCR ===
        ocr_start = time.time()
        ocr_results = Parallel(n_jobs=6, backend="threading")(
            delayed(self._ocr_single_crop)(crop) for crop in crops_for_ocr
        )
        ocr_time = time.time() - ocr_start

        # OCR 결과 원래 위치에 재매핑
        texts = [None] * len(boxes)
        for idx, text in zip(indices_for_ocr, ocr_results):
            texts[idx] = text

        ui_elements = [
            {
                "bbox": [x1, y1, x2, y2],
                "text": text
            }
            for (x1, y1, x2, y2, _), text in zip(boxes, texts)
        ]

        total_time = time.time() - total_start

        return ui_elements, {
            "yolo_time_sec": round(yolo_time, 3),
            "ocr_time_sec": round(ocr_time, 3),
            "total_time_sec": round(total_time, 3),
            "num_elements": len(ui_elements)
        }


if __name__ == "__main__":
    test_image = os.path.join(os.path.dirname(__file__), "../../app/sample/image/example5.png")
    detector = UIDetector(gpu=False)
    for i in range(5):
        print("\n----- 테스트 실행", i+1, "-----\n")

        elements, stats = detector.extract(test_image)
        print(stats)
        print("\n-----------------------\n")
    
