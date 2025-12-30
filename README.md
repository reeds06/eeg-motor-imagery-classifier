# eeg-motor-imagery-classifier
**Overview:**
EEG motor imagery classification pipeline using **MNE** and **scikit-learn**.
The goal is to classify **left-hand vs. right-hand motor imagery** from EEG recordings using **Common Spatial Patterns (CSP)** for features extraction and **Linear Discriminant Analysis (LDA)** for classification.

This project focusses on building a classical BCI pipeline rather than end-to-end deep learning, with emphasis on signal processing and cross-subject evaluation.


**Dataset:**
This project uses **BCI Competition IV Dataset 2a **, which contains EEG recordings from nine subjects performing four motor imagery tasks.

Format: GDF
Channels: 22 EEG + 3 EOG
Sampling Rate: 250 Hz
Tasks used in this project: **Left hand vs. right hand imagery**

Dataset link: https://www.bbci.de/competition/iv/

**Pipleline:**
The clasification pipeline is as follows, reflecting a standards motor imagery BCI workflow:

1. **Data Loading**
   - EEG Data is loaded from GDF files using mne.io.read_raw_gdf
   - EOG channels are removed to focus exclusively on EEG signals

2. **Preprocessing**
  - Band-pass filtering is applied between 8-30 Hz to isolate mu and beta ryhtms associated with motor imagery.
  - Events are extracted from annotation markers provided in the dataset.
  - Epochs are created from -0.1 to 4.0 s relative to motor imagery cue onset
  - Baseline correction is applied using the pre-cue interval (-0.1 to 0.0 s)

3. **Epoch Selection**
  - Only left-hand and right-hand motor imagery trials are retained for binary classification.
  - Each trail is represented as an epoch spanning from -0.1 to 4.0 s relative to the motor imagery cue.

4. **Feature Extraction**
   - Common Spacial Patterns (CSP) is used to extract features that emphasize difference in signal variance between left         and right-hand motor imagery.
   - The log-variance of a fixed number of CSP components is used as the feature representation.
  
5. **Classification**
   - A Linear Discriminant Analysis (LDA) classifier is trained on the CSP features to distinguish between left and right-        hand motor imagery.
6. **Evaluation**
   - Performance is evaluated using 5-fold stratified cross-validation to limit testing bias.
  
  
**Results:**
    - Using the CSP + LDA pipeline on left vs right-hand motor imagery trials, the model achieved an average cross-               validated accuracy of ~82%.

**Limitations**
  - The model is trained on only individual subjects, meaning that it has not been tested for cross-subject generalization.
  - No advanced artificat removal besides EOG channel exclusion was applied, so eye blinks or other noise could affect          performance.
  - Accuracy is estimated via cross-validation, and not an independent test set, so there may be impacts of overfitting.
