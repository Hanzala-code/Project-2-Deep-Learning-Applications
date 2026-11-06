# Week 8: Project 2 API (Ship It)

A FastAPI service that serves my Week 6 image model (chest X-ray, NORMAL vs PNEUMONIA) and Week 7 text evidence (PII masking, field extraction, urgent terms) through three endpoints.

## Endpoints

| Endpoint | Method | What it does |
|----------|--------|--------------|
| `/health` | GET | Reports status, whether the model is loaded, the class names, and the POSITIVE class |
| `/predict` | POST | Takes an image (JPEG or PNG) and returns label, class probabilities, and latency in ms |
| `/analyze` | POST | Takes an image and a note; returns the prediction, the masked note, extracted fields, urgent terms, a REFER or ROUTINE decision, reasons, and a disclaimer |

Settings (in `app.py`, overridable by environment variables): `MODEL_PATH=week6_model.pt`, `POSITIVE=PNEUMONIA`, `THRESHOLD=0.5`, `MAX_BYTES=5 MB`, `URGENT=['fever', 'breath', 'cough', 'chest pain', 'spreading', 'crack']`.

## How to run

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn python-multipart pillow pytest httpx numpy
uvicorn app:app --reload
```

Then open `http://127.0.0.1:8000/docs` and use "Try it out", or call it from a second terminal (PowerShell, hence `curl.exe`):

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -F "file=@sample.jpeg;type=image/jpeg"
curl.exe -X POST http://127.0.0.1:8000/analyze -F "file=@sample.jpeg;type=image/jpeg" -F "note=Cough and fever since Monday. Call 0300-1234567"
curl.exe -X POST http://127.0.0.1:8000/predict -F "file=@app.py;type=text/plain"
```

Example `/predict` response:

```json
{"label":"PNEUMONIA","probabilities":{"NORMAL":0.007,"PNEUMONIA":0.993},"latency_ms":42.8}
```

## Tests

Run with `pytest -q test_api.py`. Result:

```
8 passed, 1 warning in 8.41s
```

The eight tests cover: `/health` (model loaded), `/predict` schema and probabilities summing to 1, parity between the API and calling the model directly (within 1e-3), 415 for a wrong file type, 413 for an oversized file, 400 for a corrupt image, 422 for a missing file, and `/analyze` masking the phone number and CNIC with no leak anywhere in the response and a REFER decision.

## Latency benchmark

Run with `python bench.py` against a running server (`uvicorn app:app`, no `--reload`), 50 sequential requests to `/predict`.

**Hardware:** AMD Ryzen 5 7430U (with Radeon Graphics), 7.4 GB RAM (as reported by Windows), Windows, CPU-only PyTorch, 1 uvicorn worker.

```
n=50  p50=56 ms  p95=69 ms  max=166 ms
Throughput with 1 worker ~ 17.0 requests/s
```

**Little's law.** L = lambda x W. With one worker, L = 1 request in the system, and W = 0.056 s at my p50, so lambda = 1 / 0.056 = about 17.9 requests per second. Over a day: 17.9 x 86,400 = about 1.5 million requests per day. This is an upper bound with the worker 100% busy all day; real traffic arrives in peaks, so I would plan for a fraction of it.

**Reading the numbers.** p95 is only 13 ms above p50, so latency is stable. The 166 ms maximum is probably a first-request or warm-up effect, but I have not verified that. The benchmark sends requests one at a time, so it measures a single client, not concurrent load.

## Bad inputs I tried beyond the tests

- **Empty file:** I uploaded a completely empty file named `empty.jpg` with content type `image/jpeg`. The API returned HTTP 400 with `{"detail":"File is not a readable image"}`. The content-type check passed because of the label, but PIL could not decode zero bytes, and the error was caught and returned as a 400 instead of a 500. This differs from the corrupt-image test, which sends the bytes `b'not an image'`.
- **Oversized file:** a 6 MB file labelled `image/jpeg` returned `{"detail":"Image larger than 5 MB"}` (413).
- **Wrong type:** uploading `app.py` as `text/plain` returned `{"detail":"Upload a JPEG or PNG image"}` (415).

## API key (Task A.6)

With the `API_KEY` environment variable set, a request to `/predict` without the `X-API-Key` header returned 401 `{"detail":"Invalid or missing API key"}`. With the correct header it returned the normal prediction. `/health` stays open. The key is read from the environment and is never committed to the repository.

## Check Your Understanding

**1. A client uploads a PDF to `/predict`. Which status code should it get, and why not 500?**
415 Unsupported Media Type. The mistake is the client's: it sent a file type the API does not accept. A 500 means the server itself broke, which is wrong here because the server worked as designed, and it would hide the real cause and look like a bug in monitoring. The API checks the content type first and returns a clear message.

**2. Why is the model loaded in `lifespan` rather than inside the endpoint?**
Loading a checkpoint and building the network is slow compared with one prediction. Inside the endpoint, every request would pay that cost again and latency would jump. `lifespan` loads the model once when the server starts and keeps it in memory for all requests, which is why `/predict` takes tens of milliseconds.

**3. p50 = 40 ms and p95 = 400 ms. What does that tell you, and what would you check?**
Typical requests are fast, but about 1 in 20 is ten times slower, and the average would hide that while users feel the tail. I would check whether heavy work is blocking the event loop so some requests queue behind others, whether a few inputs are much larger than the rest, whether other processes are competing for the CPU or the laptop is throttling, and whether warm-up or garbage-collection pauses are involved. I would first look at which requests are slow before changing anything.

**4. 500,000 requests a day, peak 5 times the average, 25 requests per second per worker. How many workers?**
Average rate = 500,000 / 86,400 = about 5.8 requests/s. Peak = 5 x 5.8 = about 28.9 requests/s. Workers = 28.9 / 25 = 1.16, so 2 workers, plus some headroom in practice.

**5. Why can model inference inside an `async def` endpoint slow down other requests?**
An `async def` endpoint runs on the single event-loop thread. Model inference is CPU-heavy and does not yield control, so while it runs the loop cannot serve anything else, even a quick `/health` call, and other requests wait. `run_in_threadpool` moves the inference to a worker thread so the loop stays free. The lab's own testing showed worst-case `/health` latency tripling without it.

**6. How many communication channels does a 4-person team have?**
n(n - 1) / 2 = 4 x 3 / 2 = 6 channels.

## Limits

This is decision support only. The model's output must not be used as a diagnosis; a qualified human decides, as the `disclaimer` field in every `/analyze` response says.

# Project-2-Deep-Learning-Applications

## Week 7: Text and Multimodal AI
- Text classifier: SMS Spam Collection, 5,572 texts (13.4% spam), macro-F1 0.974 (word 1-2 grams) vs 0.982 (char 2-5 grams); spam precision/recall 0.979/0.933 (word) vs 0.986/0.953 (char)
- Top positive / negative features: call, txt, free, text, reply, claim / my, me, ok, that, it, but (some, like "to" and "your", look like style shortcuts)
- NER on my domain (clinical notes): out-of-the-box held-out F1 0.600 -> with entity ruler 0.839 (16 gold entities, so small-sample; rules written from 5 sentences, scored on the other 5)
- Regex fields extracted: CNIC, mobile, amount, date, email, invoice no., plus temperature, SpO2, age, dose, MRN; false positives tested: 4 domain negatives, all passed ("Ward 12 F block", "Stage 4 disease", "12 mgmt office", "MRN 123456789, SpO2 1920%")
- OCR: CER raw 0.062 / 0.059 / 0.143 -> after correction rules 0.000 on all three test pairs; invoice OCR 0.000 raw and after (clean synthetic image)
- Pipeline: image model + text evidence -> JSON record (example below)
- Privacy: CNIC, phone, and email masked before logging

Example record (chest X-ray model + clinical note):
{
  "image": "person100_bacteria_475.jpeg",
  "image_probabilities": {"NORMAL": 0.007, "PNEUMONIA": 0.993},
  "note_masked": "Child aged 4, fever for 3 days and fast breath. Seen at PIMS Islamabad on 12/09/2026. Guardian mobile [PHONE].",
  "text_evidence": {
    "entities": [["4","DATE"],["3 days","DATE"],["PIMS","ORG"],["12/09/2026","DATE"]],
    "fields": {"date": ["12/09/2026"]},
    "urgent_terms": ["fever", "breath"]
  },
  "decision": "REFER",
  "reasons": ["image model: P(PNEUMONIA) = 0.99", "note mentions: fever, breath"],
  "disclaimer": "Decision support only; a qualified human decides."
}
## Model card
**Model:** ResNet-18 pretrained on ImageNet, with a new 2-class head (NORMAL, PNEUMONIA). Input: chest X-ray converted to grayscale, copied to 3 channels, resized to 224 x 224 and normalised with ImageNet mean and std.

**Training:** two phases with Adam, weighted cross-entropy (weights NORMAL 1.935, PNEUMONIA 0.674), batch size 32. Phase 1 trained the head only (1,026 trainable parameters) for 2 epochs at lr 1e-3. Phase 2 also fine-tuned the last block (layer4) for 3 epochs at lr 1e-4 for the block and 1e-3 for the head. The best epoch was chosen by validation macro-F1 (0.954). Seed 42. Saved as week6_model.pt.

**Intended use:** triage that prioritises a radiologist's queue for children's chest X-rays, working at a threshold of 0.405 chosen on validation for recall of at least 0.95.

**Out of scope:** final diagnosis, adults, other age groups, other hospitals or scanners, and any setting where PPV at the local prevalence has not been checked.

**Metrics (test, used once):** macro-F1 0.914; accuracy 0.923; AUC 0.978; sensitivity 0.995 and specificity 0.786 at threshold 0.405; PPV 0.886 at 62.5% prevalence and 0.197 at 5%.

**Known failure modes:** 45 of 234 normal X-rays flagged as pneumonia at the default threshold (50 at 0.405). Grad-CAM on false positives highlights borders, shoulders and corners rather than lungs (shortcut suspected).

## Datasheet

**Motivation:** collected by Kermany et al. (2018) to develop and test AI for diagnosing pediatric pneumonia from chest X-rays; the Kaggle copy is by Paul Mooney.

**Composition:** 5,856 images in the folders (5,216 train, 624 test, 16 val), 2 classes; 1,341 NORMAL and 3,875 PNEUMONIA in train (ratio 2.9); filenames carry patient ids (1.83 images per patient); 62 exact duplicate files, all within one split.

**Collection:** retrospective clinical X-rays of children aged 1 to 5 from one hospital in Guangzhou. The time period and the imaging devices are not stated in the dataset description.

**Labelling:** diagnoses graded by expert physicians (two per image, with the evaluation set checked by a third). How disagreements were handled is not described on the Kaggle page.

**Uses:** research and education; not for diagnosis or for other populations.

## Week 6: Pediatric chest X-ray pneumonia screening with ResNet-18

**Who benefits:** radiologists and clinics triaging children's chest X-rays where radiologists are scarce.

**Why it matters:** a missed pneumonia delays treatment, while too many false flags overload the radiologist queue.

**Dataset:** Chest X-Ray Images (Pneumonia), Paul Mooney on Kaggle, from Kermany et al. (2018); license CC BY 4.0; 5,856 images (5,216 train, 624 test, 16 val folders), 2 classes; labels by expert physicians. Children aged 1 to 5 from one hospital in Guangzhou.

**Data audit:** imbalance ratio 2.9; exact duplicates 62 (all within one split: 50 in train, 12 in test; none across train and test); test images with a near-duplicate in train 0 of 624; split by patient (StratifiedGroupKFold), 0 shared groups; train 4,354, val 862, test 624.

**Imbalance handling:** weighted loss, weights NORMAL 1.935 and PNEUMONIA 0.674.

**Results (test, used once):** macro-F1 0.914; recall NORMAL 0.808 and PNEUMONIA 0.992; sensitivity 0.995 and specificity 0.786 at threshold 0.405 (set on validation); AUC 0.978; PPV 0.886 at test prevalence and 0.197 at 5% prevalence.

**Grad-CAM:** on correct predictions the model looks at the lung fields; on the three false positives inspected it looks at borders, shoulders and corners. Shortcut risk: suspected, checked so far only by inspection (next step: border-crop and re-score).

**Limitations:** pediatric patients from one hospital, 1.83 images per patient, a single test set, and Grad-CAM is coarse. Not for final diagnosis.

----------------------------------------------------------------------------------------------------------------------------------------------------

## Week 5: Scene Classification with Transfer Learning

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

