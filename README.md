# HSBAL: A Heterogeneous Selector–Based Active Learning Approach for Performance Prediction of Configurable Software Systems

This drive releases the code and data for the HSBAL model.

## Dependencies

numpy==1.26.4
pandas==2.2.3
scikit-learn==1.5.2
modAL-python==0.4.2.1
xgboost==2.1.0
joblib==1.4.2

## Directories

+ `datasets`: dataset including the performance data of the 15 subject systems.
+ `results`: the prediction results of HSBAL on the 15 subject systems.

## Usage

We test our approach in four subject, to switch the subjects, please modify the subject name in main function. There are 15 software systems that users can evaluate: x264, BDBJ, lrzip, vp9, polly, Apache, BDBC, SQL, WGet, LLVM, Dune, hipacc, hsmgp, javagc, sac. 
