import nltk,random

class GenderApp(object):
    def __init__(self):
        names_sample = nltk.corpus.names
        self.names = [(name,'male') for name in names_sample.words('male.txt')] + [(name,'female') for name in names_sample.words('female.txt')]
        random.shuffle(self.names)
        self.feature_list = [(GenderApp.gender_features_part2(name.lower()),gender) for name,gender in self.names]
        self.training_dataset = self.feature_list[:4000]
        self.testing_dataset = self.feature_list[4000:]
        self.classifier = nltk.NaiveBayesClassifier.train(self.training_dataset)
        
    def error_analysis(self):
        self.train_set = self.feature_list[2000:]
        self.test_set = self.feature_list[1000:2000]
        self.error_checking_dataset = self.names[:1000]
        self.classifier = nltk.NaiveBayesClassifier.train(self.train_set)
        self.errors = []
        for name,gender in self.error_checking_dataset:
            predicted_gender = self.check_gender_for_errors(name)
            if gender != predicted_gender:
                self.errors.append((name,gender,predicted_gender))
        print("Total errors found: ",len(self.errors))
        print("Accuracy: ",100-((len(self.errors)/len(self.error_checking_dataset)) * 100))
        for name,actual_gender,predicted_gender in self.errors:
            print('Name: ',name,'\tActual Gender: ',actual_gender,'\tPredicted Gender: ',predicted_gender)
        
    def Checking_Accuracy(self):
        print("Accuracy: ",nltk.classify.accuracy(self.classifier,self.training_dataset)*100)
        
    def Check_Gender(self,name):
        print('Name: ',name, 'Gender is: ',self.classifier.classify(GenderApp.gender_features_part2(name)))
        
    @staticmethod
    def gender_features_part2(word):
        word = str(word).lower()
        features = dict()
        features['first_letter'] = word[0]
        features['last_letter'] = word[-1]
        for char in 'abcdefghijklmnopqrstuvwxyz':
            features['count '+ char] = word.count(char)
            features['has '+ char] = char in word
        return features
    
    def check_gender_for_errors(self,name):
        return self.classifier.classify(GenderApp.gender_features_part2(name.lower()))
    
    def show_most_informative_features(self,n=10):
        self.classifier.show_most_informative_features(n)
    


if __name__ == '__main__':
    app = GenderApp()
    app.Check_Gender("apurv")
    app.Checking_Accuracy()
    app.show_most_informative_features()
    app.error_analysis()
#    for key,value in GenderApp.gender_features_part2('Sam').items():
#        print(key,value)