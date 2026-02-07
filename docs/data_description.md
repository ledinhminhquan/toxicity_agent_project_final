# Data Description Document

## Data source
This project uses a public toxicity dataset mirrored on Hugging Face from the
**Jigsaw Toxic Comment Classification Challenge** (Wikipedia talk page comments).

Dataset name (default in configs):
- `thesofakillers/jigsaw-toxic-comment-classification-challenge`

## Licensing
- The dataset card indicates redistribution under **CC0**, while the underlying comment text
  originates from Wikipedia content under **CC BY-SA 3.0**.
Always verify terms in the dataset card and your organization's compliance requirements.

## Size & languages
- Primarily English comments.
- Multi-label annotations for 6 toxicity categories.

## Labels
- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate (mapped to `identity_attack` for Detoxify compatibility)

## Preprocessing
- Minimal normalization (whitespace normalization).
- No text augmentation by default.
- Optional negative downsampling configurable in `configs/train.yaml`.

## Splits
- If dataset provides labeled splits, use them.
- Otherwise, create train/val/test with a fixed seed:
  - train: 90%
  - val: 5%
  - test: 5%

## Known limitations & biases
- Toxicity datasets are prone to **identity term bias**: mentioning certain identities can be
  incorrectly predicted as toxic.
- Labels reflect annotator perceptions and may encode cultural bias.
- The dataset contains offensive content; access and storage should be controlled.

Mitigation in this project:
- Use Detoxify "unbiased" model as a baseline.
- Track fairness metrics across identity mentions (conceptual plan in docs).
- Human-in-the-loop review for borderline cases.
