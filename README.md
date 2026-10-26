# visionAI

Hands-on projects from **비전 AI와 비즈니스 (Vision AI & Business)** — Kookmin University, AI Big Data Convergence Management.

Each folder covers one computer-vision **task**, built with Hugging Face `transformers` on Google Colab (T4 GPU).
The common domain is **medical imaging and documents**, with a focus on **explainability (XAI)**: checking *why* a model decides, not just *how accurately*.

| Task | Week | Data | Model | Highlights |
|---|---|---|---|---|
| [Image Classification](./image-classification) | 5 | Chest X-ray Pneumonia | ConvNeXt-Tiny | Transfer learning, sensitivity/specificity, Grad-CAM → shortcut learning found |
| [Image Segmentation](./image-segmentation) | 6 | Chest X-ray Pneumonia | chest-x-ray-basic (U-Net) | Lung & heart masks, CTR, lung-only retraining → shortcut moved to mask shape |
| [OCR](./ocr) | 7 | Hospital lab reports (MedOCR-Vision) | Florence-2-base | Lab value extraction + H/L flags, human-review safety layer, evidence maps |
