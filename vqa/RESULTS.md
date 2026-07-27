# VQA results

All values below are frozen seed-2026 results. Full category rows and raw
responses are in each control’s `results/` directory.

## Static scene recognition

Primary image-only accuracy:

| Model | Accuracy |
|---|---:|
| Gemini 3.1 Pro Preview | 97.3% |
| GPT-5 | 95.0% |
| Qwen 3.6 Plus | 95.3% |

The JSON-only control scored 98.7%, 97.8%, and 83.5%, respectively.

## Coordinate localization

All models returned valid coordinates for all 132 requested object centers.

| Model | Mean error | Median error | Within 100 px |
|---|---:|---:|---:|
| Gemini 3.1 Pro Preview | 2.7 px | 2.0 px | 100.0% |
| GPT-5 | 42.5 px | 38.4 px | 97.7% |
| Qwen 3.6 Plus | 30.4 px | 21.1 px | 97.7% |

On targets more than 60 pixels from both screen dividers, combined left/right
and upper/lower classification accuracy was 100.0% for Gemini and 98.8% for
both GPT and Qwen.

## Qualitative temporal recognition

| Model | 32 images | 32 JSON states |
|---|---:|---:|
| Gemini 3.1 Pro Preview | 98.5% | 96.6% |
| GPT-5 | 94.6% | 95.1% |
| Qwen 3.6 Plus | 98.0% | 97.1% |

All three models recovered upward versus downward orange-tool motion with 100%
accuracy in both representations except for the two questions in one Qwen
image call rejected by the provider's image filter. The primary Qwen image
score conservatively counts those two questions wrong; accuracy among valid
provider responses was 99.0%.

## Interpretation

These results support a calibrated claim: the tested models can parse the
task-relevant objects, use the benchmark coordinate convention, and recover
dominant motion from the supplied feedback. They do not prove perfect
perception, and they do not answer the separate action-rounding question;
coordinate replay/jitter is the appropriate control for that issue.
