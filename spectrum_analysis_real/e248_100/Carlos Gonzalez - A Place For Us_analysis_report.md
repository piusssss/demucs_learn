# Audio Separation Analysis Report

**Model**: e248_100

**Track**: Carlos Gonzalez - A Place For Us

**Analysis Time**: 2025-12-25T00:08:31.427946

---

## Drums

**Duration**: 250.22 seconds

**Sample Rate**: 44100 Hz

### 1. Energy Distribution

| Frequency Band | Real Energy | Model Energy | Ratio | Status |
|---|---|---|---|---|
| Sub-bass (20-60 Hz) | 1.97e+06 | 1.56e+06 | 0.789 | ⚠️ Under-extraction |
| Bass (60-250 Hz) | 2.68e+07 | 2.02e+07 | 0.752 | ⚠️ Under-extraction |
| Low-mid (250-500 Hz) | 2.85e+06 | 1.54e+06 | 0.541 | ⚠️ Under-extraction |
| Mid (500-2000 Hz) | 1.83e+06 | 7.78e+05 | 0.426 | ⚠️ Under-extraction |
| High (2000-8000 Hz) | 1.10e+06 | 5.47e+05 | 0.495 | ⚠️ Under-extraction |
| Total | 3.50e+07 | 2.54e+07 | 0.725 | ⚠️ Under-extraction |

### 2. Spectral Similarity

- **Overall Correlation**: 0.9108 (1.0 = perfect)
- **Cosine Similarity**: 0.9124 (1.0 = perfect)

**Band-wise Correlations**:

- Sub-bass (20-60 Hz): 0.8373 ⚠️
- Bass (60-250 Hz): 0.9277 ✅
- Low-mid (250-500 Hz): 0.8371 ⚠️
- Mid (500-2000 Hz): 0.7628 ⚠️
- High (2000-8000 Hz): 0.8022 ⚠️

### 3. Error Energy (Pseudo-SDR)

| Frequency Band | Pseudo-SDR (dB) | Error Ratio | Quality |
|---|---|---|---|
| Sub-bass (20-60 Hz) | 4.31 | 0.3704 | ⚠️ Fair |
| Bass (60-250 Hz) | 7.25 | 0.1884 | 👍 Good |
| Low-mid (250-500 Hz) | 3.28 | 0.4704 | ⚠️ Fair |
| Mid (500-2000 Hz) | 2.14 | 0.6111 | ⚠️ Fair |
| High (2000-8000 Hz) | 3.09 | 0.4904 | ⚠️ Fair |
| Total | 5.74 | 0.2667 | 👍 Good |

### 4. Silence Leakage

- **Silence Frames**: 2155 / 21553 (10.0%)
- **Leakage Ratio**: 0.1267 (lower is better)
- **Silence Energy Ratio**: 0.0008 (0.08% of total output)
- **Status**: ⚠️ Moderate leakage

### 5. Temporal Alignment (Onset Detection)

- **Real Onsets**: 955
- **Model Onsets**: 791
- **Matched Onsets**: 776 (±50ms)
- **Precision**: 0.9810 (how many model onsets are correct)
- **Recall**: 0.8126 (how many real onsets are detected)
- **F1-Score**: 0.8889
- **Status**: ✅ Excellent temporal alignment

### 6. Spectral Divergence

- **KL Divergence**: 0.059990 (0 = identical)
- **JS Divergence**: 0.015958 (0 = identical)

### 7. Dynamic Range

| Metric | Real | Model | Difference |
|---|---|---|---|
| Dynamic Range (dB) | 93.96 | 40.70 | -53.27 |
| Peak Amplitude | 0.6037 | 0.5539 | -0.0499 |
| RMS Amplitude | 0.0454 | 0.0385 | -0.0069 |
| Crest Factor (dB) | 22.47 | 23.16 | 0.68 |

---

## Bass

**Duration**: 250.22 seconds

**Sample Rate**: 44100 Hz

### 1. Energy Distribution

| Frequency Band | Real Energy | Model Energy | Ratio | Status |
|---|---|---|---|---|
| Sub-bass (20-60 Hz) | 1.75e+06 | 1.62e+06 | 0.921 | ✅ Good |
| Bass (60-250 Hz) | 6.85e+06 | 4.56e+06 | 0.666 | ⚠️ Under-extraction |
| Low-mid (250-500 Hz) | 4.01e+05 | 5.77e+04 | 0.144 | ⚠️ Under-extraction |
| Mid (500-2000 Hz) | 1.57e+05 | 1.27e+04 | 0.081 | ⚠️ Under-extraction |
| High (2000-8000 Hz) | 5.59e+04 | 2.68e+03 | 0.048 | ⚠️ Under-extraction |
| Total | 9.23e+06 | 6.69e+06 | 0.726 | ⚠️ Under-extraction |

### 2. Spectral Similarity

- **Overall Correlation**: 0.7730 (1.0 = perfect)
- **Cosine Similarity**: 0.7743 (1.0 = perfect)

**Band-wise Correlations**:

- Sub-bass (20-60 Hz): 0.8342 ⚠️
- Bass (60-250 Hz): 0.7497 ⚠️
- Low-mid (250-500 Hz): 0.4646 ❌
- Mid (500-2000 Hz): 0.2554 ❌
- High (2000-8000 Hz): 0.2035 ❌

### 3. Error Energy (Pseudo-SDR)

| Frequency Band | Pseudo-SDR (dB) | Error Ratio | Quality |
|---|---|---|---|
| Sub-bass (20-60 Hz) | 4.08 | 0.3908 | ⚠️ Fair |
| Bass (60-250 Hz) | 2.66 | 0.5421 | ⚠️ Fair |
| Low-mid (250-500 Hz) | 0.16 | 0.9638 | ⚠️ Fair |
| Mid (500-2000 Hz) | -0.31 | 1.0750 | ❌ Poor |
| High (2000-8000 Hz) | -0.16 | 1.0379 | ❌ Poor |
| Total | 2.28 | 0.5911 | ⚠️ Fair |

### 4. Silence Leakage

- **Silence Frames**: 2081 / 21553 (9.7%)
- **Leakage Ratio**: 0.2663 (lower is better)
- **Silence Energy Ratio**: 0.0053 (0.53% of total output)
- **Status**: ⚠️ Moderate leakage

### 5. Temporal Alignment (Onset Detection)

- **Real Onsets**: 418
- **Model Onsets**: 930
- **Matched Onsets**: 389 (±50ms)
- **Precision**: 0.4183 (how many model onsets are correct)
- **Recall**: 0.9306 (how many real onsets are detected)
- **F1-Score**: 0.5772
- **Status**: ⚠️ Poor temporal alignment

### 6. Spectral Divergence

- **KL Divergence**: 0.178481 (0 = identical)
- **JS Divergence**: 0.050820 (0 = identical)

### 7. Dynamic Range

| Metric | Real | Model | Difference |
|---|---|---|---|
| Dynamic Range (dB) | 85.41 | 33.94 | -51.47 |
| Peak Amplitude | 0.3446 | 0.2444 | -0.1001 |
| RMS Amplitude | 0.0233 | 0.0195 | -0.0038 |
| Crest Factor (dB) | 23.39 | 21.94 | -1.44 |

---

## Other

**Duration**: 250.22 seconds

**Sample Rate**: 44100 Hz

### 1. Energy Distribution

| Frequency Band | Real Energy | Model Energy | Ratio | Status |
|---|---|---|---|---|
| Sub-bass (20-60 Hz) | 3.14e+05 | 1.38e+05 | 0.439 | ⚠️ Under-extraction |
| Bass (60-250 Hz) | 5.65e+06 | 4.88e+06 | 0.863 | ⚠️ Under-extraction |
| Low-mid (250-500 Hz) | 1.91e+07 | 1.75e+07 | 0.916 | ✅ Good |
| Mid (500-2000 Hz) | 1.86e+07 | 1.78e+07 | 0.956 | ✅ Good |
| High (2000-8000 Hz) | 5.15e+06 | 4.14e+06 | 0.803 | ⚠️ Under-extraction |
| Total | 4.99e+07 | 4.49e+07 | 0.900 | ⚠️ Under-extraction |

### 2. Spectral Similarity

- **Overall Correlation**: 0.9160 (1.0 = perfect)
- **Cosine Similarity**: 0.9205 (1.0 = perfect)

**Band-wise Correlations**:

- Sub-bass (20-60 Hz): 0.7716 ⚠️
- Bass (60-250 Hz): 0.7838 ⚠️
- Low-mid (250-500 Hz): 0.9033 ✅
- Mid (500-2000 Hz): 0.8728 ⚠️
- High (2000-8000 Hz): 0.9478 ✅

### 3. Error Energy (Pseudo-SDR)

| Frequency Band | Pseudo-SDR (dB) | Error Ratio | Quality |
|---|---|---|---|
| Sub-bass (20-60 Hz) | 3.92 | 0.4055 | ⚠️ Fair |
| Bass (60-250 Hz) | 3.06 | 0.4940 | ⚠️ Fair |
| Low-mid (250-500 Hz) | 6.65 | 0.2164 | 👍 Good |
| Mid (500-2000 Hz) | 4.72 | 0.3373 | ⚠️ Fair |
| High (2000-8000 Hz) | 8.30 | 0.1478 | 👍 Good |
| Total | 5.38 | 0.2895 | 👍 Good |

### 4. Silence Leakage

- **Silence Frames**: 2156 / 21553 (10.0%)
- **Leakage Ratio**: 0.0645 (lower is better)
- **Silence Energy Ratio**: 0.0004 (0.04% of total output)
- **Status**: ✅ Minimal leakage

### 5. Temporal Alignment (Onset Detection)

- **Real Onsets**: 535
- **Model Onsets**: 818
- **Matched Onsets**: 432 (±50ms)
- **Precision**: 0.5281 (how many model onsets are correct)
- **Recall**: 0.8075 (how many real onsets are detected)
- **F1-Score**: 0.6386
- **Status**: 👍 Good temporal alignment

### 6. Spectral Divergence

- **KL Divergence**: 0.005712 (0 = identical)
- **JS Divergence**: 0.001448 (0 = identical)

### 7. Dynamic Range

| Metric | Real | Model | Difference |
|---|---|---|---|
| Dynamic Range (dB) | 75.71 | 37.81 | -37.90 |
| Peak Amplitude | 0.3967 | 0.3651 | -0.0315 |
| RMS Amplitude | 0.0540 | 0.0513 | -0.0026 |
| Crest Factor (dB) | 17.33 | 17.04 | -0.28 |

---

## Vocals

**Duration**: 250.22 seconds

**Sample Rate**: 44100 Hz

### 1. Energy Distribution

| Frequency Band | Real Energy | Model Energy | Ratio | Status |
|---|---|---|---|---|
| Sub-bass (20-60 Hz) | 1.02e+03 | 1.17e+05 | 114.833 | ⚠️ Over-extraction |
| Bass (60-250 Hz) | 9.32e+06 | 6.64e+06 | 0.713 | ⚠️ Under-extraction |
| Low-mid (250-500 Hz) | 2.59e+07 | 2.08e+07 | 0.803 | ⚠️ Under-extraction |
| Mid (500-2000 Hz) | 2.88e+07 | 2.00e+07 | 0.695 | ⚠️ Under-extraction |
| High (2000-8000 Hz) | 1.44e+06 | 1.05e+06 | 0.729 | ⚠️ Under-extraction |
| Total | 6.60e+07 | 4.96e+07 | 0.751 | ⚠️ Under-extraction |

### 2. Spectral Similarity

- **Overall Correlation**: 0.9188 (1.0 = perfect)
- **Cosine Similarity**: 0.9201 (1.0 = perfect)

**Band-wise Correlations**:

- Sub-bass (20-60 Hz): 0.1777 ❌
- Bass (60-250 Hz): 0.8837 ⚠️
- Low-mid (250-500 Hz): 0.9386 ✅
- Mid (500-2000 Hz): 0.9098 ✅
- High (2000-8000 Hz): 0.8722 ⚠️

### 3. Error Energy (Pseudo-SDR)

| Frequency Band | Pseudo-SDR (dB) | Error Ratio | Quality |
|---|---|---|---|
| Sub-bass (20-60 Hz) | -20.62 | 115.4549 | ❌ Poor |
| Bass (60-250 Hz) | 5.35 | 0.2917 | 👍 Good |
| Low-mid (250-500 Hz) | 8.13 | 0.1538 | 👍 Good |
| Mid (500-2000 Hz) | 6.50 | 0.2240 | 👍 Good |
| High (2000-8000 Hz) | 4.74 | 0.3359 | ⚠️ Fair |
| Total | 6.66 | 0.2159 | 👍 Good |

### 4. Silence Leakage

- **Silence Frames**: 2156 / 21553 (10.0%)
- **Leakage Ratio**: 0.0663 (lower is better)
- **Silence Energy Ratio**: 0.0003 (0.03% of total output)
- **Status**: ✅ Minimal leakage

### 5. Temporal Alignment (Onset Detection)

- **Real Onsets**: 371
- **Model Onsets**: 480
- **Matched Onsets**: 285 (±50ms)
- **Precision**: 0.5938 (how many model onsets are correct)
- **Recall**: 0.7682 (how many real onsets are detected)
- **F1-Score**: 0.6698
- **Status**: 👍 Good temporal alignment

### 6. Spectral Divergence

- **KL Divergence**: 0.038978 (0 = identical)
- **JS Divergence**: 0.012377 (0 = identical)

### 7. Dynamic Range

| Metric | Real | Model | Difference |
|---|---|---|---|
| Dynamic Range (dB) | 76.81 | 41.59 | -35.23 |
| Peak Amplitude | 0.5762 | 0.5142 | -0.0621 |
| RMS Amplitude | 0.0624 | 0.0540 | -0.0084 |
| Crest Factor (dB) | 19.31 | 19.58 | 0.27 |

---

## Summary

### Overall Assessment

- **Drums**: 60% 👍 Good
- **Bass**: 20% ❌ Poor
- **Other**: 70% 👍 Good
- **Vocals**: 60% 👍 Good

### Generated Visualizations

- Spectrum comparison: 4 images
- Waveform comparison: 4 images
- Mask comparison: 4 images
- Energy envelope: 4 images
- Band energy evolution: 4 images
- Phase consistency: 4 images
- Comprehensive comparison: 1 image

**Total**: 25 images
