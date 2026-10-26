# Image Segmentation — Lung & Heart Segmentation on Pediatric Chest X-rays

> **Week 6** · 비전 AI와 비즈니스 · Pretrained segmentation + using masks to test the Week 5 shortcut problem

## TL;DR
- Applied a pretrained **lung + heart segmentation model** to pediatric chest X-rays with **no fine-tuning**. Both lungs and the heart were found in **100% of 624 test images**.
- Mask-based **CTR and lung area** revealed **class-dependent acquisition bias**: pneumonia images had a higher CTR and smaller lungs, suggesting shallow inspiration and supine positioning rather than true cardiomegaly.
- Retraining the [Week 5 classifier](../image-classification) on **lung-only images** did **not** fix the shortcut. Specificity fell 0.611 → 0.517, and Grad-CAM moved to the **mask outline**. Removing pixels ≠ removing information.

## Setup
| | |
|---|---|
| **Dataset** | [`hf-vision/chest-xray-pneumonia`](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia) — same as Week 5 (pediatric AP, 1–5 y, CC BY 4.0). **No ground-truth masks** → inference only |
| **Segmentation model** | [`ianpan/chest-x-ray-basic`](https://huggingface.co/ianpan/chest-x-ray-basic) — EfficientNetV2-S + U-Net, 22.1M params, right lung / left lung / heart. No license listed; research use only |
| **Classifier** | [`facebook/convnext-tiny-224`](https://huggingface.co/facebook/convnext-tiny-224) — same seed, split and hyperparameters as Week 5 |

## Pipeline
1. Load the segmentation model. `from_pretrained` fails on transformers v5 (custom `CXRModel` code), so the config and `model.safetensors` are loaded directly (0 missing / 0 unexpected keys).
2. Predict right lung / left lung / heart masks and overlay them.
3. Without GT masks, check quality via **detection rate, lung area ratio and CTR** (cardiothoracic ratio).
4. Use the masks to black out everything outside the lungs and crop (`CROP_MODE="mask"`).
5. Retrain ConvNeXt on the lung-only images and compare with Week 5 (metrics, confusion matrix, Grad-CAM).

## Part 1 — Segmentation results

![segmentation overlay](images/segmentation_overlay.png)
*Red = right lung · Green = left lung · Blue = heart*

| Test set (624) | NORMAL (234) | PNEUMONIA (390) |
|---|---|---|
| Both lungs + heart detected | 100% | 100% |
| Lung area ratio (mean ± SD) | 0.276 ± 0.050 | 0.256 ± 0.039 |
| CTR (mean ± SD) | 0.468 ± 0.042 | 0.513 ± 0.049 |

Inference: 94 ms/image on T4 · Lung crops for all 5,856 images: ~6 min, 0 failures.

![lung area and CTR](images/lung_area_ctr.png)

- The adult PA CTR cut-off of **0.5 does not apply to children**. The heart is relatively larger, and AP projection magnifies it. Pediatric reference lines are shown instead: infant ~0.60, child ~0.55.
- By pediatric standards most images in **both** classes are within normal range. The CTR gap more likely reflects **inspiration depth and positioning** than cardiomegaly.
- CTR here is computed from masks, not measured by a radiologist. It is useful for **group comparison only**.

**Lowest lung-area cases** are not segmentation failures. They are infants imaged with a wide field of view (abdomen, arms), several with ECG leads and lines.
![lowest lung area](images/lowest_lung_area.png)

## Part 2 — Lung-only classification

![lung crop](images/lung_crop.png)

| Metric (test, 624) | Week 5 (full image) | Week 6 (lung only) |
|---|---|---|
| Accuracy | 0.851 | 0.817 |
| F1 (macro) | 0.824 | 0.776 |
| Sensitivity (pneumonia) | 0.995 | **0.997** |
| Specificity (normal) | 0.611 | **0.517** |
| FN / FP | 2 / 91 | 1 / 113 |
| Training time (3 epochs) | ~5.8 min | ~2.6 min |

| Learning curve | Confusion matrix |
|---|---|
| ![learning curve](images/learning_curve.png) | ![confusion matrix](images/confusion_matrix.png) |

- Val loss fell every epoch (0.158 → 0.062 → 0.051), with **no overfitting**, unlike Week 5.
- But **test loss was 0.971 vs val 0.051**. Validation from the training distribution is even more over-optimistic than in Week 5.

**Grad-CAM**
![gradcam](images/gradcam.png)
- NORMAL prediction → **inside the right lung** (perihilar) ✅
- PNEUMONIA predictions → **the blacked-out heart region, lung bases and image corners** ❌

## Findings
- **The segmentation transfers well to children**, even though the model was not trained on this population.
- **Removing pixels does not remove information.** The mask outline still encodes heart width (≈ CTR) and lung shape (inspiration), so the classifier used the **mask shape** as a new shortcut instead of looking for consolidation.
- **The root cause is data bias.** NORMAL and PNEUMONIA differ in acquisition conditions (posture, inspiration, lines/ECG leads), and preprocessing alone cannot remove that.
- **Lost information:** standard lung masks exclude retrocardiac and sub-diaphragmatic lung, so lesions there are cut out.

## Business takeaway
The segmentation model works out-of-the-box and can support **pre-reading tasks**:
- lung-field highlighting
- automatic CTR measurement
- acquisition-quality checks, such as poor inspiration

It could also serve as a **data-quality tool**, flagging class-dependent acquisition bias before a medical AI dataset is used for training. Segmentation-based preprocessing alone, however, does not make a diagnostic model trustworthy. XAI and external validation are still required.

## Next steps
- [ ] Compare `CROP_MODE="bbox"` (keeps context, no outline cue)
- [ ] Make mask shape uninformative: random mask dilation/erosion, per-lung resizing
- [ ] Attention constraint — penalize Grad-CAM outside the lung mask ("right for the right reasons")
- [ ] Lesion-level labels (e.g., RSNA pneumonia bounding boxes) → object detection

## Run it
Open [`image_segmentation_chest_xray_lung_heart.ipynb`](./image_segmentation_chest_xray_lung_heart.ipynb) in Colab → Runtime: **T4 GPU** → Run all (~15 min).

## References
- Kermany, D. et al. (2018). *Identifying medical diagnoses and treatable diseases by image-based deep learning.* Cell 172(5).
- Ronneberger, O. et al. (2015). *U-Net: Convolutional networks for biomedical image segmentation.* MICCAI.
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* ICCV.
- Geirhos, R. et al. (2020). *Shortcut learning in deep neural networks.* Nature Machine Intelligence.
