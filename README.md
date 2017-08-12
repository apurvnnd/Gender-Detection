# Gender-Detection
Detection of Gender using Name of the said person

The first file 'gender-detection-1.py' uses basic nltk library, importing and using the names_sample from corpus.
Dividing the Test and Train set.
It predicts the Gender of the name using just the last letter of the given name, when trained with the training set provided from names_sample.
Naive Bayes Classifier is used for the above mentioned procedure.

The second file 'gender-detection-oop.py' also uses basic nltk library, importing and using the names_sample from corpus.
It is done in a more neat manner, using OOPs in python.
Training and Test set are divided.
A seperate Training, Test and Error set are also formed for detecting the number of errors during prediction.
The Prediction is done using the first and last letter,
and also using the number of occurences of letters of the name of the person.
Naive Bayes Classifier is also used for the same. 
