# Project-2-Deep-Learning-Applications
## Project 2, Week 5: Scene Classification with Transfer Learning

- Dataset: Intel Image Classification, 6 classes
  - Train: 11,929 images | Val: 2,105 images | Test: 3,000 images
  - Classes: buildings, forest, glacier, mountain, sea, street

- Model: ResNet-18 pretrained on ImageNet (11,179,590 total parameters)

- Feature extraction (3,078 trainable parameters — 0.03% of network):
  - Epoch 1: val acc 0.894 | Epoch 2: val acc 0.899 | Epoch 3: val acc 0.901
  - Best validation accuracy: 0.901

- Fine-tuning layer4 (8,396,806 trainable parameters):
  - Epoch 4: val acc 0.926 | Epoch 5: val acc 0.930 | Epoch 6: val acc 0.928
  - Best validation accuracy: 0.930 (+0.029 over feature extraction)

- Test accuracy (used once): 0.935
  - Strongest class: forest (recall 0.987, F1 0.990)
  - Weakest class: glacier (recall 0.866, F1 0.893)
  - Most confused pair: glacier → mountain (57 errors), mountain → glacier (31 errors)
  - Second confused pair: buildings → street (36 errors)

- Most instructive error: buildings images shot at street level with
  visible roads were predicted as street with 0.98–0.99 confidence —
  the model is not wrong so much as the two classes genuinely overlap
  in these images, revealing a labelling ambiguity in the dataset

- Next: my own domain in Week 6 — PCB defect Detection

---

### Week 6 Proposal

## Week 6 Proposal: PCB Defect Detection

**Domain:** Printed Circuit Board (PCB) surface defect detection
using deep learning image classification.

**Problem:** Manual visual inspection of PCBs is slow, expensive, and
error-prone — a single missed defect can cause an entire electronic
device to fail in the field. Automating this with a trained CNN would
allow manufacturers to inspect boards at production-line speed with
consistent accuracy.

**Who benefits:** Electronics manufacturers, quality control engineers,
and PCB assembly lines — particularly relevant in Pakistan's growing
electronics manufacturing sector. A reliable defect detector reduces
waste, lowers rework costs, and prevents faulty products from reaching
consumers.

**Candidate dataset:** PCB Defect Dataset (Kaggle, maknight0/pcb-defects
or the DeepPCB dataset) — contains labelled images of six common PCB
defect types: missing hole, mouse bite, open circuit, short circuit,
spur, and spurious copper. Approximately 1,500 to 2,000 images across
6 classes, with a permissive license suitable for academic use.

Dataset link: https://www.kaggle.com/datasets/akhatova/pcb-defects

- Classes: 6 (missing_hole, mouse_bite, open_circuit,
  short, spur, spurious_copper)
- Images: ~1,500 training images
- Input: microscope/X-ray PCB surface images, RGB

**Why transfer learning fits:** PCB defect images share low-level
features (edges, texture discontinuities, colour anomalies) with
natural images, so ImageNet-pretrained ResNet weights provide a strong
starting point even for this industrial domain. Week 5's pipeline
(feature extraction then fine-tuning layer4) will be reused directly,
with augmentation adjusted to avoid flips that would make a "spur"
defect look like a different defect type.

---

### How to Run

```bash
# Install dependencies
pip install torch torchvision

# Run the notebook on Kaggle with GPU T4 enabled
# Dataset: Intel Image Classification (puneet6060)
```

📓 [Week 5 Notebook — Transfer Learning](https://www.kaggle.com/code/hanzalabhatti/week-5-transfer-learning)
