# LANL Earthquake Prediction
The goal of the challenge is to capture the physical state of the laboratory fault and how close it is from failure from a snapshot of the seismic data it is emitting. You will have to build a model that predicts the time remaining before failure from a chunk of seismic data.
Additional info:

- The input is a chunk of 0.0375 seconds of seismic data (ordered in time), which is recorded at 4MHz, hence 150'000 data points, and the output is time remaining until the following lab earthquake, in seconds.

- The seismic data is recorded using a piezoceramic sensor, which outputs a voltage upon deformation by incoming seismic waves. The seismic data of the input is this recorded voltage, in integers.

- Both the training and the testing set come from the same experiment. There is no overlap between the training and testing sets, that are contiguous in time.

- Time to failure is based on a measure of fault strength (shear stress, not part of the data for the competition). When a labquake occurs this stress drops unambiguously.

- The data is recorded in bins of 4096 samples. Withing those bins seismic data is recorded at 4MHz, but there is a 12 microseconds gap between each bin, an artifact of the recording device.


## TODOs

1. Controllare se le pause siano effettivamente ogni 4095 osservazioni (vedi notebook explore)


## Kernels

### Feature Engineering

* https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction
* https://www.kaggle.com/michael422/spectrogram-convolution
* https://www.kaggle.com/nikitagribov/analysis-function-for-seismic-signal-data 