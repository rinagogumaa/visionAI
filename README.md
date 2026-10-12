# visionAI

Hands-on projects from **비전 AI와 비즈니스 (Vision AI & Business)** — Kookmin University, AI Big Data Convergence Management.

Each folder covers one computer-vision **task**, built with Hugging Face `transformers` on Google Colab (T4 GPU).
Medical imaging (pediatric chest X-ray) is used as the common domain, with a focus on **explainability (XAI)**: checking *why* a model decides, not just *how accurately*.

| Task | Week | Data | Model | Highlights |
|---|---|---|---|---|
| [Image Classification](./image-classification) | 5 | Chest X-ray Pneumonia | ConvNeXt-Tiny | Transfer learning, sensitivity/specificity, Grad-CAM → shortcut learning found |
| Image Segmentation | 6 | Chest X-ray Pneumonia | chest-x-ray-basic (U-Net) | *coming soon* — lung masking to fix Week 3's shortcut problem |
