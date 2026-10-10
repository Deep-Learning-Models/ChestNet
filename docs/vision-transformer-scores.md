# Vision Transformer scores and improved training workflow

The original notebook trained successfully, stopped early at epoch 9, and restored epoch 4. Its output is complete. The earlier chat statement that it contained only three epochs was based on a truncated inspection and was incorrect.

The dataset used by that run was a sampled subset: `configs/base.yaml` sets `max_images_per_class: 1500`, and `build_subset()` in `src/preprocessing/splits.py` randomly selects images without replacement, starting with the rarest selected class. The union has 7,685 images from 4,751 patients. Multi-label overlap means the union is not eight times 1,500 and some class totals exceed the target. The run uses 5,400 training, 1,109 validation, and 1,176 test images.

## What the original results show

All entries below are copied from the saved notebook output, rounded as printed. They are scores against NIH labels on its particular test subset.

| Class | Positive test images | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Atelectasis | 230 | 0.302 | 0.813 | 0.440 | 0.775 |
| Cardiomegaly | 255 | 0.496 | 0.831 | 0.622 | 0.896 |
| Effusion | 231 | 0.387 | 0.593 | 0.468 | 0.780 |
| Infiltration | 258 | 0.373 | 0.519 | 0.434 | 0.686 |
| Mass | 246 | 0.536 | 0.459 | 0.495 | 0.799 |
| Nodule | 229 | 0.316 | 0.655 | 0.427 | 0.711 |
| Pneumonia | 200 | 0.337 | 0.490 | 0.399 | 0.692 |
| No Finding | 234 | 0.342 | 0.739 | 0.468 | 0.765 |

The reported macro ROC-AUC is **0.7630**, macro F1 is **0.4691**, and inference throughput is **16.93 ms per image**. The class scores in the table are rounded; averaging them may differ slightly from the macro scores computed from unrounded values. The timing includes `model.predict()` and its input pipeline; it is an amortized throughput measurement rather than isolated latency for a single GPU prediction.

### What each score means

| Score | Meaning | How to interpret it here |
|---|---|---|
| Support | Number of images with that label | An image can count toward several labels. These counts do not sum to the number of distinct test images. |
| Precision | TP / (TP + FP) | Atelectasis precision 0.302 means about 30% of its positive predictions carry that NIH label. The remaining positive predictions disagree with the label. |
| Recall | TP / (TP + FN) | Atelectasis recall 0.813 means it finds about 81% of images labeled positive, while missing about 19%. |
| F1 | 2 × precision × recall / (precision + recall) | Atelectasis F1 remains 0.440 because precision is much lower than recall. High recall alone cannot produce a high F1. |
| ROC-AUC | Ranking discrimination across decision thresholds | Cardiomegaly AUC 0.896 means strong positive-versus-negative ranking on this test split. It does not mean 89.6% accuracy. Chance ranking is 0.5; perfect ranking is 1. |
| Macro average | Each class gets equal weight | A weak class matters as much as a strong one. Macro F1 averages class F1 values; it is not necessarily the F1 calculated from macro precision and macro recall. |
| Binary accuracy in training logs | Fraction of individual label decisions that match | Most image/class pairs are negative, so high accuracy can coexist with weak detection. It differs from requiring every label on an image to be correct. |
| Weighted BCE loss | Penalizes incorrect continuous scores, with extra weight for positives | Lower loss is better under the same weighting, but it is not a percentage and need not track thresholded F1. |

TP, FP, FN, and TN describe agreement with the dataset's labels. NIH labels were derived from reports, so label disagreement is not itself proof of a radiological error. The dataset paper describes the report-derived labels and weak supervision: [Wang et al.](https://arxiv.org/abs/1705.02315).

### Why the scores have this pattern

**Overfitting is visible in the logs.** At epoch 4, training AUC was 0.7925 and validation AUC was 0.7640. By epoch 9, training AUC had reached 0.9116 but validation AUC had fallen to 0.7450. Training loss fell from 0.8661 to 0.5840 while validation loss rose from 0.9275 to 1.1210. Early stopping responded appropriately and restored the earlier checkpoint. The best validation AUC of 0.7640 and test AUC of 0.7630 are close, although they measure different patient groups.

**The threshold and weighted loss help explain high recall with low precision.** The old model used a fixed 0.5 threshold and positive weights around four. Positive weighting encourages higher scores for positives and can increase positive predictions. In an idealized weighted-BCE model with weight `w`, the optimal score for an underlying probability `p` is `w*p / (1-p+w*p)`. With `w=4`, a score cutoff of 0.5 corresponds to `p=0.2` in that idealized calculation. The trained model is not guaranteed to obey this calibration, but it explains why assuming the weighted sigmoid score is an ordinary disease probability can be misleading. Validation threshold tuning may improve F1; its actual effect must be measured.

**Class differences are measured, but their causes are not established.** Cardiomegaly has the strongest ranking and F1; Infiltration and Pneumonia have the weakest ranking. Mass has the highest precision but misses more than half its labeled positives at the chosen threshold. Nodule finds more positives but also produces many false positives. Possible explanations include signal size, overlapping findings, image resolution, label noise, and training sample diversity. The notebook output alone cannot determine which explanation caused an individual class's performance. Error review and controlled experiments are needed.

**Sampling changes what the scores represent.** The original subset has roughly balanced positive counts, unlike the naturally imbalanced full data. Class prevalence and which negatives are included affect precision, F1, and average precision. ROC-AUC is less directly driven by prevalence, but it still depends on the distribution and difficulty of positives and negatives. A full-data experiment uses different patients and label distributions, so a difference from 0.7630 or 0.4691 does not by itself prove that an architecture or training change caused improvement.

## What the new notebook adds

The new file is `notebooks/vision-transformer-improved.ipynb`. It uses PyTorch and timm and saves a `.pt` inference package. The existing TensorFlow `.h5` checkpoint has a different architecture implementation and is not loaded into the new model.

1. Audit every image in the attached metadata; report image count, patient count, bytes and GiB; reject missing, corrupt, duplicate, or conflicting records.
2. Keep all metadata images, with no balanced image downsampling. Images containing only unselected diseases are negatives for the selected outputs; `No Finding` remains its explicit label.
3. Create and save fixed patient splits: approximately 70% training, 10% epoch selection, 5% threshold tuning, and 15% test. Check zero patient and image overlap. These are custom splits.
4. Start from an [ImageNet pretrained ViT-B/16](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k), with its normalization and an eight-logit classifier.
5. Train the classifier first, then the last four transformer blocks, then the full encoder at a lower learning rate. Use mild augmentation, AdamW, warmup/cosine scheduling, clipping, and early stopping.
6. Use one or two CUDA GPUs, mixed precision, and gradient accumulation. Save the best model and an epoch checkpoint that includes optimizer and random state.
7. Choose class thresholds on the separate tuning validation group. Report test results both at 0.5 and at those thresholds.
8. Report ROC-AUC, average precision, precision, recall, F1, specificity, confusion counts, macro/micro metrics, label/exact-match accuracy, Brier error, and contradictory `No Finding` decisions.
9. Compute confidence intervals by resampling patients, create calibration and generalization plots, and save subgroup results and concrete error examples.
10. Save the inference model with class order, preprocessing, and thresholds; verify reloading on actual images; export an archive and an automatically generated explanation of the measured scores.

The default training budget is seven hours, excluding auditing and final evaluation, with a maximum staged schedule of 2 + 5 + 8 epochs. Hitting that budget is recorded explicitly and retains the latest complete epoch for resumption. Early stopping and time limits can mean fewer epochs run. The configured settings are a starting experiment; they are not a completed hyperparameter search.

## Additional scores in the new notebook

| Score | Meaning |
|---|---|
| Average precision (AP) | Summarizes precision over recall levels using continuous scores. Compare it with each class's prevalence. It is not trapezoidal PR-AUC. |
| Specificity | TN / (TN + FP): fraction of negatives correctly rejected. |
| Micro F1 | Pools TP/FP/FN across classes before calculating F1. |
| Subset accuracy | Fraction of images whose entire eight-label vector is correct. |
| Brier mean | Average squared difference between scores and 0/1 labels. Lower is better; weighted training may impair calibration. |
| Conflict rate | Fraction with simultaneous `No Finding` and disease predictions. |
| Patient bootstrap interval | Uncertainty from test-patient sampling conditional on the selected model and thresholds. It does not measure variation between training runs. |

Metric definitions follow the [scikit-learn metrics documentation](https://scikit-learn.org/stable/api/sklearn.metrics.html) and [average precision documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
