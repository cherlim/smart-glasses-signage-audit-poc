# Smart Glasses Signage Audit POC

This is a minimum proof of concept for using smart-glasses-style first-person observations to perform signage auditing.

## What this POC demonstrates

It does **not** yet perform real computer vision. Instead, it simulates the output of a smart-glasses vision pipeline:

- detected sign text
- sign location
- bounding-box size
- visibility duration
- OCR confidence
- contrast
- occlusion

The audit engine then generates signage audit metrics.

## Files

- `app.py` — Streamlit dashboard
- `building.json` — synthetic indoor building graph
- `scenarios.json` — signage scenarios
- `requirements.txt` — Python dependencies

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Research framing

The research concept is:

> Smart glasses as a first-person data collection device for automated wayfinding signage auditing.

The first phase is not about GNNs. It is about proving that wearable visual data can be transformed into measurable signage audit indicators.

## Example audit metrics

- Sign visibility score
- Sign readability score
- Decision-point coverage
- Destination guidance detection
- Sign density
- Problem sign count

## Next upgrade

Replace simulated sign observations with real video processing:

```text
Smart glasses / phone video
      ↓
Sign detection
      ↓
OCR
      ↓
Audit metrics
      ↓
Dashboard
```