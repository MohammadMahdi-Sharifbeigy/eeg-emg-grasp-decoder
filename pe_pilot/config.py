"""Settings of the prediction-error pilot (no training; every analysis reads the raw WS files and P{p}_AllLifts)."""

CONFIG = {
    "data_dir": "data/way-eeg/raw",
    "results_dir": "results/pe_pilot",
    "participants": list(range(1, 13)),
    "fs": 500.0,                                    # EEG, force and the EMG envelope after resampling
    "eeg": {
        "band": (0.5, 30.0),                        # slow potentials and ERP
        "alpha": (8.0, 13.0), "beta": (15.0, 25.0),
        "reject_mad": 3.0,                          # reject a lift when its peak-to-peak in the touch window (central
                                                    # channels) exceeds the participant median + 3 robust SD; a fixed
                                                    # uV threshold fails here: the median lift is already ~300 uV
    },
    "emg": {"band": (20.0, 450.0), "lowpass": 10.0},
    "touch_window": (-0.5, 1.5),                    # s around tBothDigitTouch (error analysis)
    "led_baseline": (-0.5, 0.0),                    # s around LEDOn: rest baseline of the band power (dB)
    "pre_contact": (-0.6, -0.05),                   # s around touch: reach window of the expectation decoder
    "pre_cue": (-0.5, 0.0),                         # s around LEDOn: carry-over control of the decoder
    "peripheral": ["Fp1", "Fp2", "F7", "F8", "T7", "T8", "TP9", "TP10", "PO9", "PO10", "O1", "Oz", "O2"],
    "error_roi": ["C3", "Cz", "CP1", "CP5", "FC1", "FC5"],   # left sensorimotor: the right hand lifts
    "error_window": (0.2, 0.6),                     # s after touch: a-priori window of the single-trial error size
    "n_permutations": 1000,                         # cluster tests over participants
    "n_label_shuffles": 200,                        # decoder null per participant
    "cluster_p": 0.05,
}

KIN_LF, KIN_GF = 38, 41                             # ws.names.kin[36:] = IndLF ThuLF LF IndGF ThuGF GF ...
WEIGHT_GRAMS = {1: 165, 2: 330, 4: 660}
