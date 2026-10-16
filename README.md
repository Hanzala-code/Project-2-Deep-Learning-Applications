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

- Next: my own domain in Week 6 — [your domain idea here]

---

### Week 6 Proposal

[Write one paragraph here: your chosen domain, the problem it solves,
who benefits, the candidate dataset link, number of classes and images]

---

### How to Run

```bash
# Install dependencies
pip install torch torchvision

# Run the notebook on Kaggle with GPU T4 enabled
# Dataset: Intel Image Classification (puneet6060)
```

📓 [Week 5 Notebook — Transfer Learning]([your Kaggle link here])
