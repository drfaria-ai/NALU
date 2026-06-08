# NALU-PTBR: Neuro-Affective Language Understanding Dataset and Framework

**NALU-PTBR** is a Brazilian Portuguese neuro-affective language dataset and single-file Transformer framework for interpretable affective, psychological, behavioural, clinical-support and psychometric-oriented language understanding.

The framework is designed for **screening-oriented research and clinical-support applications only**. It does not provide diagnosis, psychiatric assessment, medical advice or autonomous clinical decision-making.

## Short abstract

NALU-PTBR combines Brazilian Portuguese affective language, sentiment resources, translated emotion datasets, conversational mental-health data, psychometric questionnaire-derived supervision, clinically relevant linguistic patterns, silver annotations and controlled neuro-affective augmentation. The associated `nalu_framework.py` code trains a multi-head BERTimbau-based architecture that jointly predicts sentiment, emotions, psychological cues, behavioural intentions, mental-state risk indicators, clinical-language markers, Russell valence-arousal dimensions and normalized psychometric targets associated with PHQ-9, GAD-7 and DASS dimensions. A deterministic neuro-symbolic interpretation layer provides calibrated, auditable and non-diagnostic explanations in Portuguese.

## Repository contents

```text
README.md
nalu_framework.py
NALU-PTBR.csv                  # dataset file, not always included in public repositories
license.txt -> for academic use only.
```

## Dataset overview

NALU-PTBR contains:

| Property | Value |
|---|---:|
| Samples | 205,109 |
| Columns | 48 |
| Supervised outputs | 45 |
| Text language | Brazilian Portuguese |
| Psychometric-supervised rows | 42,257 |
| Maximum sequence length used in the paper | 160 WordPiece tokens |
| Encoder used in the framework | `neuralmind/bert-base-portuguese-cased` |

The corpus was constructed by harmonizing and merging affective, sentiment, conversational mental-health and psychometric resources into a unified Brazilian Portuguese representation space. The paper describes the dataset as integrating public affective datasets, Portuguese sentiment corpora, MentalChat16K, anxiety/depression indicator datasets, DASS-related datasets, university student mental-health indicators, additional Kaggle/Mendeley mental-health datasets, translated emotion resources and controlled neuro-affective augmentation.

Synthetic and silver-labeled samples are used as complementary enrichment to improve coverage of rare affective, psychological, behavioural and clinical-support language patterns. Psychometric supervision is grounded in questionnaire-associated data and is represented as normalized screening-oriented regression targets.

## Dataset file name

Use the following canonical dataset filename:

```text
NALU-PTBR.csv
```

If your local dataset has another filename, rename it to `NALU-PTBR.csv` or pass the path explicitly using `--dataset`.

## Dataset schema

Each row corresponds to one text sample with dense multi-task supervision.

### Core columns

| Column | Description |
|---|---|
| `sample_id` | Unique sample identifier |
| `text` | Brazilian Portuguese input text |

### Russell affective-space regression

| Column | Range | Description |
|---|---:|---|
| `valence_russell` | [-1, 1] | Continuous valence target |
| `arousal_russell` | [-1, 1] | Continuous arousal target |

### Sentiment head

Multi-class one-hot target with 3 outputs:

| Column |
|---|
| `sentiment_positive` |
| `sentiment_neutral` |
| `sentiment_negative` |

### Emotion head

Multi-label target with 9 outputs:

| Column |
|---|
| `emotion_joy` |
| `emotion_sadness` |
| `emotion_anger` |
| `emotion_fear` |
| `emotion_disgust` |
| `emotion_surprise` |
| `emotion_calm` |
| `emotion_distress` |
| `emotion_neutral` |

### Psychological cue head

Multi-label target with 11 outputs:

| Column |
|---|
| `stress_pressure_cue` |
| `hopelessness_cue` |
| `self_devaluation_cue` |
| `loss_of_control_cue` |
| `social_withdrawal_cue` |
| `irritability_cue` |
| `cognitive_overload_cue` |
| `guilt_shame_cue` |
| `resilience_coping_cue` |
| `self_focus_cue` |
| `empathy_cue` |

### Behavioural-intention head

Multi-label target with 6 outputs:

| Column |
|---|
| `hostility` |
| `toxicity` |
| `seeking_help_intent` |
| `avoidance_intent` |
| `coping_intent` |
| `aggressive_intent` |

### Risk-language head

Multi-label target with 4 outputs:

| Column |
|---|
| `anxiety_related_language` |
| `depression_related_language` |
| `crisis_danger_text` |
| `mental_state_risk_flag` |

### Clinical-language marker head

Multi-label target with 5 outputs:

| Column |
|---|
| `grief_loss_language` |
| `sleep_disturbance_cue` |
| `relationship_distress_cue` |
| `caregiver_burden_cue` |
| `panic_attack_language` |

### Psychometric-oriented regression head

Continuous normalized targets in [0, 1]:

| Column | Description |
|---|---|
| `phq9_score_norm` | Normalized PHQ-9-oriented target |
| `gad7_score_norm` | Normalized GAD-7-oriented target |
| `dass_depression_norm` | Normalized DASS Depression-oriented target |
| `dass_anxiety_norm` | Normalized DASS Anxiety-oriented target |
| `dass_stress_norm` | Normalized DASS Stress-oriented target |
| `psychometric_mask` | 1 when psychometric supervision is valid; 0 otherwise |

The psychometric outputs are **language-derived screening-oriented estimates**. They should not be interpreted as clinical scores or diagnostic predictions.

## Label groups and heads

NALU uses eight prediction heads:

| Head | Type | Outputs |
|---|---|---:|
| Sentiment | Multi-class classification | 3 |
| Emotion | Multi-label classification | 9 |
| Psychological cues | Multi-label classification | 11 |
| Behavioural intentions | Multi-label classification | 6 |
| Risk language | Multi-label classification | 4 |
| Clinical-language markers | Multi-label classification | 5 |
| Psychometric-oriented estimates | Continuous regression | 5 |
| Russell affective space | Continuous regression | 2 |

Total supervised outputs: **45**.

## Framework pipeline

The code in `nalu_framework.py` implements the complete NALU pipeline:

```text
Input text
  -> BERTimbau tokenizer
  -> Shared BERTimbau Transformer encoder
  -> Eight task-specific heads
  -> Multi-task optimization
  -> Threshold calibration for multi-label heads
  -> Neuro-symbolic contradiction suppression
  -> Contextual risk calibration
  -> Deterministic Portuguese interpretation
  -> Reports, metrics, predictions and exported thresholds
```

The framework uses:

- BERTimbau Base: `neuralmind/bert-base-portuguese-cased`
- Maximum length: 160 tokens
- Sentiment loss: weighted cross-entropy
- Multi-label heads: asymmetric focal loss
- Russell and psychometric targets: Smooth L1 regression loss
- Psychometric masking: rows without valid psychometric targets are ignored only for the psychometric loss
- Checkpoint selection: composite multi-head validation score
- Interpretation: deterministic neuro-symbolic explanation layer

## Installation

Create an environment with Python 3.10 or newer.

```bash
pip install torch transformers pandas numpy scikit-learn scipy tqdm matplotlib
```

For GPU training, install the PyTorch build appropriate for your CUDA setup.

## Training

```bash
python nalu_framework.py train \
  --dataset ./NALU-PTBR.csv \
  --output_dir ./nalu_runs/main \
  --epochs 15 \
  --batch_size 16 \
  --eval_batch_size 32 \
  --max_length 160
```

Useful options:

```bash
--learning_rate 2e-5
--head_learning_rate 1e-4
--dropout 0.2
--gradient_checkpointing
--bootstrap_iterations 1000
```

## Prediction from text

```bash
python nalu_framework.py predict \
  --checkpoint_dir ./nalu_runs/main \
  --texts "Eu estou me sentindo muito pressionado, mas estou tentando lidar melhor."
```

## Prediction from file

The input file may be `.txt`, `.csv` or `.json`. For CSV files, use `--text_column` to indicate the text column.

```bash
python nalu_framework.py predict_file \
  --checkpoint_dir ./nalu_runs/main \
  --input_file ./examples/texts.csv \
  --text_column text
```

## Baseline comparison

The script also supports lightweight baseline experiments:

```bash
python nalu_framework.py baseline \
  --dataset ./NALU-PTBR.csv \
  --output_dir ./nalu_runs/baselines \
  --baseline_models majority tfidf frozen_bertimbau frozen_xlmr
```

## Output files

A training run exports:

```text
best_model.pt
config.json
thresholds.json
validation metrics
test metrics
prediction CSV files
head-level performance tables
per-label performance tables
alert distributions
sentiment and emotion summaries
interpretation calibration files
```

## Safety and ethical use

NALU-PTBR is intended for research on affective computing, human-centred AI, educational support, wellbeing interfaces, clinical-support tools and human-robot or human-computer interaction. It should not be used as an autonomous clinical system.

Important constraints:

- Do not use the psychometric outputs as diagnosis.
- Do not use risk outputs as emergency triage without professional oversight.
- Do not deploy the model for high-stakes decisions without validation, governance and human supervision.
- Interpret outputs as probabilistic language-derived indicators.
- Follow data protection and ethical approval requirements when using sensitive text.

## Reference

If you use this dataset or framework, please cite:

```text
Diego Resende Faria, "Interpretable Neuro-Affective Language Understanding for Psychological and Psychometric Assessment Towards Clinical-Support Applications", Scientific Reports, 2026 (under review).
```

BibTeX:

```bibtex
@article{faria2026nalu,
  title   = {Interpretable Neuro-Affective Language Understanding for Psychological and Psychometric Assessment Towards Clinical-Support Applications},
  author  = {Faria, Diego Resende},
  journal = {Scientific Reports},
  year    = {2026},
  note    = {Under review}
}
```

## Contact

Dr Diego Resende Faria  
Department of Computer Science, School of Science  
Loughborough University, United Kingdom
