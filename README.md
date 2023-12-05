# Octopawn solver
### John Asencio

### Purpose of this program
This is a machine learner that can achieve at least 70% accuracy in classifying instances anomalous vs normal. The data is based off this paper https://pubmed.ncbi.nlm.nih.gov/11583923/ by Kurgan et al describes diagnosis of heart anomalies using machine learning on SPECT radiography data. 

### What I did
I implemented a naive bayes algorithm that is used by a K-fold cross validation algorithm to test the results on subsets of the data then gathers the average of all the results. The data is binarized by 1's and 0's which makes it easy to determine normal and abnormal classifications. The data is represented in CSV files, which my program takes and reads in. Each row represents a (patient visit) and with the first comma-separated field representing the class.

### How it went
At first, I could not read in the data properly. Then implementing naive bayes was not too hard but it was not giving me the expected results. I was very confused on as to why. At first I assumed it was because I was not cross validating properly, which is why I implemented K-fold cross validation. After this was done, I was still not getting proper results. I kept checking the K-fold function to make sure it was not failing. Going through multiple tests. It was finally found that I was just doing the calculation for Naive bayes wrong. I was still dividing through the total, not the total 0s and 1s which was why my percentage for true negative rate was so low. 

## How to run
python heart_anomalies.py heart-anomalies.csv