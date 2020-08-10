# Natural Language Processing system using binary logistic regression 
## feature.py
```python feature.py [args1...]``` \
Where above [args1...] is a placeholder for eight command-line arguments:train input, validation input, test input, dict input, formatted train out, formatted validation out, formatted test out, feature flag \
1. train input: path to the training input .tsv file 
2. validation input: path to the validation input .tsv file 
3. test input: path to the test input .tsv file 
4. dict input: path to the dictionary input .txt file 
5. formatted train out: path to output .tsv file to which the feature extractions on the train- ing data should be written 
6. formatted validation out: path to output .tsv file to which the feature extractions on the validation data should be written 
7. formatted test out: path to output .tsv file to which the feature extractions on the test data should be written 
8. feature flag: integer taking value 1 or 2 that specifies whether to construct the Model 1 feature set or the Model 2 feature set

## lr.py
```python lr.py [args2...]``` \
On the other hand, [args2...] is a placeholder for eight command-line arguments:formatted train input, formatted validation input, formatted test input, dict input, train out, test out, metrics out, num epoch
1. formatted train input: path to the formatted training input .tsv file
2. formatted validation input: path to the formatted validation input .tsv file 
3. formatted test input: path to the formatted test input .tsv file 
4. dict input: path to the dictionary input .txt file 
5. train out: path to output .labels file to which the prediction on the training data should be written 
6. test out: path to output .labels file to which the prediction on the test data should be written 
7. metrics out: path of the output .txt file to which metrics such as train and test error should be written 
8. num epoch: integer specifying the number of times SGD loops through all of the training data (e.g., if <num epoch> equals 5, then each training example will be used in SGD 5 times).
