def gender_features_part1(word):
    return {'last_letter':word[-1:]}

#print(gender_features_part1("sam"))



from nltk.corpus import names as names_sample
import nltk,random

names = [(name,'male') for name in names_sample.words('male.txt')] + [(name,'female') for name in names_sample.words('female.txt')]

#for name,female in names:
#    print('Name: ',name,  'Gender: ',female)

random.shuffle(names)

feature_sets = [(gender_features_part1(name.lower()),gender) for name,gender in names]

train_set = feature_sets[3000:]
test_set = feature_sets[:3000]

classifier = nltk.NaiveBayesClassifier.train(train_set)

#print(classifier.classify(gender_features_part1("tanu")))

print(nltk.classify.accuracy(classifier,test_set)*100)

print(classifier.show_most_informative_features())