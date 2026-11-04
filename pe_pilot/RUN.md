# Prediction-error pilot

Does the EEG respond when the object is heavier or lighter than expected, and does it encode the expected weight
before contact? No training: every step reads the raw WS files and `P{p}_AllLifts.mat`.

```
env/Scripts/python.exe -m pe_pilot.run                                  # all steps, all participants
env/Scripts/python.exe -m pe_pilot.run --step epochs,behaviour --participants 1-3
env/Scripts/python.exe tools/build_pe_pilot_notebook.py                 # rebuilds pe_pilot.ipynb
```

| # | step | what it does | output |
|---|---|---|---|
| 1 | `epochs` | EEG, EMG envelope and force around touch, features before contact; cached per participant | `results/pe_pilot/cache/`, `step1_epochs/counts.csv` |
| 2 | `behaviour` | weight-matched force differences (sanity check of the labels) | `step2_behaviour/` |
| 3 | `error` | cluster tests of unexpected minus expected, same object weight; EEG, EMG, force | `step3_error/` |
| 4 | `expectation` | decoder of the previous weight before contact, with EMG, hand-speed, pre-cue and peripheral controls | `step4_expectation/` |
| 5 | `learning` | EEG error size vs next-lift change of grip-force rate (exploratory) | `step5_learning/` |

The labels: `change` is `expected` (same weight as the previous lift), `heavier` or `lighter`; the first lift of a
series and lifts where the surface also changed have no label. Contrasts are taken within each current weight, so the
object is the same on both sides.

Notes on the data found while building the pilot:
- The derived grip force channel (`GF`, kin column 41) is stored negative; the pilot flips it.
- The median lift has about 300 uV peak-to-peak over two seconds of the central channels (0.5 to 30 Hz), so artifact
  rejection is relative: median + 3 robust SD per participant.
- The EMG order of the files is AD, BR, FD, ED, FDI; labels are read from the file.
