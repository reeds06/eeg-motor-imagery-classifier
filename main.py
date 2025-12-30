import mne
import matplotlib.pyplot as plt
import numpy as np


from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline



"""
Load raw EEG data from GDF file, preporcess, and return epochs of motor imagery trials
"""
def load_epochs(path):

    #read continuous EEG signal
    raw = mne.io.read_raw_gdf(path, preload=True) 

    #get rid of eog channels
    raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])

    #apply bandpass filter to focus on motor imagery frequencies
    raw.filter(l_freq=8., h_freq=30., fir_design='firwin')  # keep 8–30 Hz


    #Each row in events will contain a time in samples where the event ocurred, a placeholder, and the type of above event that it was
    """
    Events is a nx3 array
    first column: sample index
    second column: placeholder
    third column: event ID (what is the person doing in this sample)
    """

    
    events, event_dict_raw= mne.events_from_annotations(raw)
    print("Raw annotation mapping:", event_dict_raw)



    event_ids = {
        'left': 7,
        'right': 8,
        'tongue':9,
        'foot':10
    }
    
    #epoch the data (create short time windows around each event)
    epochs=mne.Epochs(raw, events, event_id=event_ids, tmin=-0.1, tmax=4.0, baseline=(-.1, 0), on_missing="warn")
    return epochs


def classify_left_vs_right(epochs):
    #select only left and right motor imagery data
    epochsLR = epochs['left', 'right']

    #extract dat and labels
    """
    (n_trials, n_channels, n_timepoints)
    """
    X = epochsLR.get_data()

    #contains labels corresponding to X
    y = epochsLR.events[:,2]
    

    

    csp = CSP(
        n_components=6,
        reg=None,
        log=True,
        norm_trace=False
    )

    lda = LinearDiscriminantAnalysis()

    pipeline = Pipeline([
        ('csp', csp),
        ('lda', lda)
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv)

    print(f"Mean accuracy: {scores.mean():.3f}")
    print(f"Std accuracy:  {scores.std():.3f}")

    return scores


def main():
    path = "/Users/reedsussman/Desktop/BCICIV_2a_gdf/A01T.gdf"

    epochs = load_epochs(path)

    # Optional sanity checks
    epochs.plot(n_epochs=5, n_channels=10)
    epochs.plot_psd(fmin=4, fmax=40)

    classify_left_vs_right(epochs)


if __name__ == "__main__":
    main()







main()
