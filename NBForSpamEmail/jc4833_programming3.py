# original performance (with the defaults) on the validation set
# Precision:0.9795918367346939 
# Recall:0.7164179104477612 
# F-Score:0.8275862068965516 
# Accuracy:0.9640933572710951

#Improved performance after tuning
# k = 1
# c = 0.00001
# Precision:0.9354838709677419 
# Recall:0.8656716417910447 
# F-Score:0.8992248062015503 
# Accuracy:0.9766606822262118

# After using stopword_file.txt
# k = 1, c = 1 is better
# Precision:0.9821428571428571 
# Recall:0.8208955223880597 
# F-Score:0.8943089430894309 
# Accuracy:0.9766606822262118


import sys
import math
import re

def extract_words(text):
    text = text.lower()
    punctuation = ['(', ')', '?', ':', ';', ',', '.','…','—','-','+','*','!', '/', '"', "'"] 
    for p in punctuation:
        text = text.replace(p,'')
    words = re.sub('(-?\d+)',"Number",text) #replace all number with 'Number'
    words = words.split(" ")
    return words


class NbClassifier(object):

    def __init__(self, training_filename,stopword_file = None):
        self.attribute_types = set()
        self.label_prior = {}    
        self.word_given_label = {}   
        self.c = 1

        self.collect_attribute_types(training_filename, 1, stopword_file)
        self.train(training_filename)          

    def collect_attribute_types(self, training_filename, k, stopword_file=None):
        attribute = {}
        filterAttribute = set()
        with open(training_filename,'r') as f:
            for line in f:
                line = line.strip()
                words = extract_words((line.split("\t"))[1])
                for w in words:
                    if w in attribute:
                        attribute[w] += 1
                    else:
                        attribute[w] = 1
                    if attribute[w] >= k:
                        filterAttribute.add(w)
        if stopword_file != None:
            with open(stopword_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line in filterAttribute:
                        filterAttribute.remove(line)
        self.attribute_types = filterAttribute

    def train(self, training_filename):
        self.label_prior = {}
        self.word_given_label = {}
        self.label_prior["ham"] = 0
        self.label_prior["spam"] = 0
        self.spamWord = {}
        self.hamWord = {}
        with open(training_filename,'r') as f:
            for line in f:
                line = line.strip()
                sentence = line.split("\t")
                label = sentence[0]
                words = extract_words(sentence[1])
                for w in words:
                    if w in self.attribute_types:
                        if label == "spam":
                            self.spamWord[w] = self.spamWord[w]+1 if w in self.spamWord else 1
                        else:
                            self.hamWord[w] = self.hamWord[w]+1 if w in self.hamWord else 1
                self.label_prior[label] += 1
        for w in self.spamWord:
            self.word_given_label[(w,"spam")] = (self.c+self.spamWord[w])/(len(self.spamWord)+self.c*(len(self.attribute_types)))
        for w in self.hamWord:
            self.word_given_label[(w,"ham")] = (self.c+self.hamWord[w])/(len(self.hamWord)+self.c*(len(self.attribute_types)))


    def predict(self, text):
        result = {}
        result["ham"] = math.log(self.label_prior["ham"]/(self.label_prior["spam"]+self.label_prior["ham"]))
        result["spam"] = math.log(self.label_prior["spam"]/(self.label_prior["spam"]+self.label_prior["ham"]))
        notham = math.log(self.c/(len(self.hamWord)+self.c*(len(self.attribute_types))))
        notspam = math.log(self.c/(len(self.spamWord)+self.c*(len(self.attribute_types))))
        for w in extract_words(text):
            result["ham"] += math.log(self.word_given_label[(w,"ham")]) if w in self.hamWord else notham
            result["spam"] += math.log(self.word_given_label[(w,"spam")]) if w in self.spamWord else notspam
        return result


    def evaluate(self, test_filename):
        precision = 0.0
        recall = 0.0
        fscore = 0.0
        accuracy = 0.0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        with open(test_filename,'r') as f:
            for line in f:
                line = line.strip()
                msg = line.split("\t")
                pred = self.predict(msg[1])
                pred_label = max(pred,key=lambda k:pred[k])
                if pred_label == "spam" and msg[0] == "spam":
                    tp += 1
                elif pred_label == "spam" and msg[0] == "ham":
                    fp += 1
                elif pred_label == "ham" and msg[0] == "spam":
                    fn += 1
                elif pred_label == "ham" and msg[0] =="ham":
                    tn += 1
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        fscore = (2*precision*recall)/(precision+recall)
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        return precision, recall, fscore, accuracy


def print_result(result):
    print("Precision:{} Recall:{} F-Score:{} Accuracy:{}".format(*result))


if __name__ == "__main__":
    if len(sys.argv) == 4:
        classifier = NbClassifier(sys.argv[1], sys.argv[3])
    else:
        classifier = NbClassifier(sys.argv[1])
    result = classifier.evaluate(sys.argv[2])
    print_result(result)

