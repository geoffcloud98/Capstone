# ECG Signal Classification: CNN vs Decision Tree vs KNN

This project compares the performance of three machine learning models. Convolutional Neural Networks (CNN), Decision Trees, and K-Nearest Neighbors (KNN), in classifying ECG (electrocardiogram) signals. The goal is to determine which model performs best on the ECG5000 dataset.


### ECG5000 
The original data set for *ECG5000* is a 20-hour long ECG downloaded from Physionet_ATM [1]. The database is BIDMC Congestive Heart Failure Database(chfdb) and the record is chf07. It was originally published by Goldberger et al. [2]. The data were pre-processed in two steps, first extracting each heartbeat and then making each heartbeat equal length using interpolation. These data were originally used by Chen et al. [3]. After that, 5000 heartbeats were randomly selected to make the current data set. Data were from a patient who has severe congestive heart failure. The class values were obtained by automated annotation.

>[1] https://physionet.org/cgi-bin/atm/ATM  
>[2] Goldberger, Ary L., et al. "PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals." Circulation 101.23 (2000): e215-e220.  
>[3] Chen, Yanping, et al. "A general framework for never-ending learning from time series streams." Data Mining and Knowledge Discovery 29.6 (2015): 1622-1664.  
>[4] http://www.timeseriesclassification.com/description.php?Dataset=ECG5000

### Models
Convolutional Neural Network (CNN)
- Input: ECG signals transformed into RGB images
- Purpose: Leverage spatial patterns for robust feature extraction
- Tool: TensorFlow/Keras

Decision Tree
- Input: Engineered features
- Purpose: Baseline interpretable model

K-Nearest Neighbors (KNN)
- Input: Raw ECG signals 
- Purpose: Distance-based benchmark

### Results
| Model         | Accuracy | F1 Score | Notes                      |
| ------------- | -------- | -------- | -------------------------- |
| CNN           | 90%    | 0.628    | Best performance overall   |
| Decision Tree | 88%    | 0.474    | Fast, interpretable        |
| KNN           | 89%    | 0.590    | Sensitive to noisy samples |

### Future Work
- Hyperparameter tuning for KNN and Decision Tree
- Test on larger or multi-lead ECG datasets
- Balance underrepresented classes using SMOTE or ADASYN
