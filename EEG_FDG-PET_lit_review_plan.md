# Literature Review Plan — EEG → FDG-PET PCA Prediction (Mayo Clinic project)

## Part 1: What the codebase actually shows

### A flag before anything else: "PCA" here means Principal Component Analysis, not the clinical syndrome

Your project brief described the outcome as a "PCA (posterior cortical atrophy pattern)." Everything in the repo — README, `train_cv.py`, `eeg_dataset.py`, `run_all_models.py` — uses "PCA" to mean **Principal Component Analysis components of the FDG-PET scan**, not the clinical posterior cortical atrophy variant of Alzheimer's. `model_data/matched_pca_vectors.parquet` stores a 150-dimensional PCA score vector per subject (1,580 subjects total), and the pipeline trains one CNN per component (`run_all_models.py`: "Model N → PC N," models 1–10 have been run). This is the classic Moeller/Habeck-style spatial-covariance-pattern approach applied to FDG-PET, not a label for the clinical PCA-AD phenotype.

This distinction changes what the lit review needs to cover, so I've built the plan around "PCA = statistical decomposition of metabolic pattern," while flagging the clinical PCA-AD literature as a secondary thread worth checking (see Section 2.2) in case the spatial loadings of your best-predicted components (PC1, PC5, PC8) turn out to resemble a posterior/parieto-occipital pattern — which would tie the two meanings together and could actually resolve the naming ambiguity into a nice observation for the paper. Worth confirming with whoever generated `matched_pca_vectors.parquet` what the PCA was run on (voxelwise SUVR image, ROI-wise summary measures) and how many components were retained/why.

### Cohort

- 1,574–1,580 subjects with paired 19-channel EEG + FDG-PET-derived PCA vectors (`non_ica_19_channels` and `orig_eeg_raw_19_channels` both contain 1,574 subject parquet files; `matched_pca_vectors.parquet` has 1,580 rows). An earlier 8-channel version of the pipeline used 1,582 subjects.
- No demographic or diagnostic fields (age, sex, diagnosis, MMSE, amyloid/tau status) exist anywhere in this repo — not in the parquet schemas, not in any notebook or script. That metadata must live in Mayo's source data outside this project folder. `hpc_setup/SETUP.md` confirms this is a Yale–Mayo collaboration (data transferred via Globus from your Mac to Yale's Bouchet HPC cluster, under PI grant `pi_am2359`), so the cohort is presumably a Mayo registry (Mayo Clinic Study of Aging is the most likely candidate given the sample size and EEG+PET pairing, but this isn't stated anywhere in the code — worth confirming before you write the Methods section).

### EEG features

- Raw scalp EEG, 19-channel standard 10-20 montage (Fp1, Fp2, F3, F4, F7, F8, Fz, C3, C4, Cz, T7, T8, P3, P4, P7, P8, Pz, O1, O2), 256 Hz sampling.
- Preprocessing (`data_processing.ipynb`): FIR bandpass 0.5–45 Hz (zero-phase, Hamming window), average reference, 10-second non-overlapping epochs (2,560 samples), per-channel z-scoring. A parallel ICA-based pipeline (Infomax ICA + `mne_icalabel`/ICLabel) auto-excludes muscle/eye/heart/line-noise components, with an amplitude-based bad-segment rejection at 500 µV in the current 19-channel version.
- The deep model consumes **raw waveform segments directly** — no hand-engineered spectral features. A separate classical baseline (`baseline_ML_model.ipynb`) computes log relative band power (delta/theta/alpha/beta/gamma via Welch PSD, 19 channels × 5 bands = 95 features) and fits RidgeCV, explicitly for comparison against the CNN.

### Modeling approach

- `EEGTemporalCNN` (`train_cv.py`): 1D CNN, three conv blocks (64→128→256 channels, kernels 128/16/8), BatchNorm, GELU, a custom concatenated avg+max pooling module, dropout 0.2, FC head to `n_outputs`. ~493K–755K params depending on channel count.
- PyTorch Lightning training: AdamW (lr 3e-4, weight decay 1e-2), MSE loss on standardized PCA targets, early stopping on subject-aggregated validation R² (patience 10).
- Each subject contributes many randomly-sampled 10-second windows per epoch; window-level predictions are averaged per subject before computing R² (the paper's primary metric).
- Evaluation: 10-fold CV stratified by PC1 quantile (20 bins, one subject per bin held out per fold → 200 unique held-out subjects total, ~1,373 always in the training pool), with an ensemble of the top-3 checkpoints per fold averaged at test time.
- A separate CNN is trained per PCA component — models 1 through 10 have been run (PC1...PC10), not a single joint multi-output model, though the architecture supports multi-output.
- `interpretability_analysis.ipynb` runs two ablation studies on a trained PC1 model: frequency-band ablation (band-stop filtering delta/theta/alpha/beta/gamma) and spatial-channel ablation (zeroing each electrode + its 3 nearest 10-20 neighbors, with topomap visualization), plus a qualitative raw-vs-ICA power spectrum comparison at T7/T8.

### Results so far

From the saved CV prediction files (`model_data/cv_results_stratified/predictions_1.parquet`...`predictions_10.parquet`, n=200 pooled held-out subjects per component):

| PC | Pooled R² |
|----|-----------|
| PC1 | 0.278 |
| PC2 | 0.131 |
| PC3 | −0.008 |
| PC4 | 0.084 |
| PC5 | 0.344 |
| PC6 | 0.046 |
| PC7 | 0.045 |
| PC8 | 0.183 |
| PC9 | 0.036 |
| PC10 | −0.020 |

Only PC1, PC5, and PC8 show a clearly-above-noise EEG-derived signal; most components are near chance. The README's headline number (R² ≈ 0.28 on PC1) matches this table. The classical Ridge/band-power baseline for PC1, run on the same 1,573-subject pool with the same 10-fold stratified scheme, gets pooled R² ≈ 0.085 (0.085 ± 0.070 across folds) — well below the CNN's 0.278, which is a good "raw waveform beats hand-crafted spectral features" result to lean on.

One thing worth reconciling before you write results: `interpretability_analysis.ipynb` contains two different runs with different baseline R² values (0.5646 on one 315-subject 80/20 split using a specific checkpoint, then 0.1846 later in the same notebook, apparently from a different checkpoint/rerun) and correspondingly different channel/frequency ablation rankings. These aren't necessarily wrong — they may reflect two different checkpoints or preprocessing variants being compared — but they shouldn't both end up in the paper without you first confirming which one is the "real" final model.

---

## Part 2: Literature review plan

### 2.1 Structure — five background areas, in the order I'd build them

**1. Clinical/biological motivation.** FDG-PET hypometabolism as an established AD biomarker; why EEG is an attractive cheap/portable surrogate. This is the framing paragraph, not deep review — a handful of well-cited sources.

**2. FDG-PET pattern-based (PCA/SSM) approaches in AD.** This is the literature your outcome variable is built on, and it's probably underrepresented in your current thinking since it's not mentioned anywhere in the codebase. Key thread: the Scaled Subprofile Model / PCA method for extracting spatial covariance ("disease-related metabolic pattern") from FDG-PET, originated by Moeller & Strother for multivariate PET analysis and extended by Eidelberg's group (well known for the Parkinson's-related pattern, PDRP) and separately for AD (an "AD-related metabolic pattern," ARMP/ADRP, work associated with Habeck, Stern, Mosconi, and others). You should confirm whether your PCA pipeline follows this SSM tradition specifically or is a more generic voxel/ROI PCA, since that determines which of these papers are direct methodological precedent vs. general background.

**2.2 Clinical posterior cortical atrophy (secondary, contingent on the naming question above).** If it turns out your best-predicted components (PC1/PC5/PC8) spatially resemble the posterior/parieto-occipital hypometabolism pattern characteristic of the clinical PCA-AD variant, this becomes directly relevant — Mayo's own group (Whitwell, Josephs, Jack) has published extensively on PCA-AD neuroimaging, and given this is Mayo data, it's worth checking whether that's a coincidence or the actual source of the "PCA" name.

**3. EEG as a biomarker in AD/MCI.** Quantitative EEG changes in AD — spectral slowing (increased delta/theta, decreased alpha/beta), reduced signal complexity, connectivity changes. Established review anchors: Jeong (2004), Babiloni's group's extensive qEEG-AD body of work, more recent computational-EEG-in-dementia reviews (e.g., Cassani et al. 2018). This grounds why delta/theta and specific scalp regions are plausible carriers of AD-relevant signal, which connects directly to your frequency-band and channel ablation results.

**4. EEG–PET / EEG–neuroimaging correlation and prediction studies.** The literature closest to directly competing with this paper — prior work correlating qEEG spectral features with regional SUVR/glucose metabolism or amyloid/tau PET, and any prior attempts (classical ML or deep learning) to predict imaging-derived phenotypes from EEG. This is where you need to search hardest, because it's what determines your novelty claim. Search explicitly for EEG-to-amyloid-PET and EEG-to-tau-PET prediction papers too, even though your outcome is FDG-PET — the modeling approaches transfer and reviewers will expect you to have looked.

**5. Methodological precedents for the pipeline itself.** Raw-waveform 1D CNNs for EEG (Schirrmeister et al.'s Deep ConvNets, EEGNet, Roy et al.'s 2019 deep-learning-for-EEG review) as architectural precedent; ICLabel (Pion-Tonachini et al. 2019) for the automated IC classification/exclusion approach; occlusion/ablation-based interpretability for EEG-DL models (channel- and frequency-masking is a form of occlusion sensitivity — cite precedent for that specific technique in EEG rather than treating it as novel); and precedent for subject-level aggregation of windowed predictions as an evaluation strategy in EEG deep learning.

### 2.2 Search strategy

Databases: PubMed/MEDLINE as primary (I have a PubMed search tool connected in this session if you want me to run any of these once you're ready), Google Scholar for citation-chasing and grey literature, IEEE Xplore for the DL-architecture side of area 5, bioRxiv/medRxiv for recent preprints in this fast-moving space (also connected here).

Suggested search strings, one per area above:

- Area 1/3: `("EEG" OR "electroencephalography" OR "quantitative EEG" OR "qEEG") AND ("Alzheimer" OR "dementia" OR "mild cognitive impairment")` — filter to reviews first, then primary spectral-marker studies.
- Area 2: `("principal component analysis" OR "scaled subprofile model" OR "spatial covariance pattern" OR "metabolic pattern") AND ("FDG-PET" OR "fluorodeoxyglucose" OR "glucose metabolism") AND "Alzheimer"`
- Area 2.2 (contingent): `"posterior cortical atrophy" AND ("FDG-PET" OR "hypometabolism")`
- Area 4 (highest priority): `("EEG" AND "PET") AND ("Alzheimer" OR "dementia") AND ("correlat*" OR "predict*")`, then separately `"EEG" AND "amyloid PET"` and `"EEG" AND "tau PET"` and `("EEG" OR "electroencephalography") AND ("deep learning" OR "convolutional neural network") AND ("PET" OR "neuroimaging") AND "predict"`
- Area 5: `("deep learning" OR "convolutional neural network") AND "EEG" AND ("Alzheimer" OR "dementia")`; separately `"ICLabel"` and `"independent component analysis" AND "EEG artifact"` for the preprocessing citations; `"raw EEG" AND "convolutional neural network" AND "regression"` for the general (non-AD) raw-waveform-CNN precedent.
- Cohort: `"Mayo Clinic Study of Aging" AND "EEG"` and `"Mayo Clinic Study of Aging" AND "FDG-PET"` — once you confirm which Mayo cohort this actually is, search for that cohort's foundational description paper (e.g., Roberts et al. 2008 for MCSA) and any prior EEG or FDG-PET sub-studies from it.

### 2.3 Organizing and screening

- One reference manager (Zotero/EndNote) with a tag per area (1–5) plus a "novelty-comparison" tag for anything from Area 4 that predicts or correlates with imaging from EEG — those are the papers your Discussion needs to explicitly position against.
- Pull recent systematic reviews per area first to map the landscape quickly, then snowball (backward and forward citation search) from anchor papers: Minoshima et al. 1997 (classic AD FDG-PET pattern), Moeller/Habeck-style SSM-PCA papers, a qEEG-AD review (Jeong 2004 or a Babiloni review), and whatever direct EEG-to-imaging-prediction papers turn up in Area 4.
- Given this is Mayo data, specifically check whether Mayo's own group has already published the FDG-PET PCA/pattern-derivation methodology as a separate paper — if so, that's the citation for how your target variable was constructed, and it sharpens your novelty claim to "the EEG-side prediction is new, the PET-side pattern is established and cited," rather than looking like you're re-deriving both halves from scratch.

### 2.4 Framing novelty

Based on what's in the repo, your defensible novelty claims are: (a) predicting continuous FDG-PET PCA-pattern scores — not diagnosis classification or amyloid positivity — directly from raw, minimally-preprocessed EEG via an end-to-end CNN; (b) a sample size (~1,574 paired subjects) that is large relative to typical EEG-PET correlation studies, which tend to run under a few hundred; (c) a head-to-head comparison against a classical band-power + Ridge baseline on the identical CV split, showing the raw-waveform model outperforms hand-crafted spectral features; and (d) per-component modeling across 10 PCA targets with frequency-band and spatial-channel ablation tied to the specific components that are actually predictable. Area 4 of the review is what will tell you whether (a) is genuinely novel or whether someone has already done EEG→FDG-PET regression with deep learning — that's the search to run first and most carefully.

### 2.5 Structuring the eventual lit review section

Recommended order for the paper itself: motivation (AD, hypometabolism, cost of PET) → FDG-PET pattern/PCA methodology background → qEEG-in-AD background → existing EEG-imaging correlation/prediction work (with an explicit gap statement closing this subsection) → DL-on-raw-EEG methodological precedent → one paragraph synthesizing the gap into your specific contribution.
