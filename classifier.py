import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

worddict = dict()
labeldict = dict()

class RNNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(RNNet, self).__init__()

        #what is the hidden size?
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

        #check out softmaxes
        self.softmax = nn.LogSoftmax(dim=1)

    #figure out what forward step actually does
    def forward(self, input, hidden):
        combined = torch.cat((input,hidden),1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)

        return output, hidden

    #need initHidden
    def initHidden(self):
        return torch.zeros(1,self.hidden_size)
        
    #what is num_flat_features?

def assembleVocab(filepath):
    print("Getting dictionary...")
    
    txtfile = open(filepath,"r",encoding="utf-8")

    wcount = 0
    lcount = 0
    for line in txtfile:
        linesplit = line.split("|||")
        stripped = linesplit[0].strip()
        if not stripped in labeldict:
            labeldict.update({stripped:lcount})
            lcount += 1
        
        words = linesplit[1].split()
        for word in words:
            if not word in worddict:
                worddict.update({word:wcount})
                wcount += 1

    labeldict.update({"<UNK>":lcount})
    worddict.update({"<UNK>":wcount})
    wcount += 1
    worddict.update({"<EOS>":wcount})
                
    wdictfile = open("wdict.txt","w+",encoding="utf-8")
    for key in worddict:
        wdictfile.write(str(key) + "\n")
    wdictfile.close()

    ldictfile = open("ldict.txt","w+",encoding="utf-8")
    for key in labeldict:
        ldictfile.write(str(key) + "\n")
    ldictfile.close()
                
    txtfile.close()

def loadVocab():
    wdictfile = open("wdict.txt","r",encoding="utf-8")
    ldictfile = open("ldict.txt","r",encoding="utf-8")
    
    wcount = 0
    lcount = 0
    
    for line in wdictfile:
        worddict[line.strip()] = wcount
        #print(line + str(worddict[line]))
        wcount += 1

    worddict["<EOS>"] = wcount
        
    for line in ldictfile:
        labeldict[line.strip()] = lcount
        lcount += 1
    
    wdictfile.close()
    ldictfile.close()
    
def getLabelLine(filepath, datatype):
    if datatype == "train":
        #Switching these two lines allows the code to generate the dictionary
        #on a first pass, then load from a file on subsequent testings.
        assembleVocab(filepath)
        #loadVocab()
            
    txtfile = open(filepath,"r",encoding="utf-8")

    features = []
    labels = []

    for line in txtfile:
        linesplit = line.split("|||")
        labels.append(labelizer(linesplit[0].strip()))
        features.append(featurizer(linesplit[1]))

    size = len(labels)
    #labeltens = torch.tensor(labels)
    #featuretens = torch.tensor(features)

    txtfile.close()
    return(features,labels,size)
    #return (featuretens,labeltens,size)

def labelizer(label):
    if label in labeldict:
        labels = [labeldict.get(label)]
    else:
        labels = [labeldict.get("<UNK>")]

    return labels

def featurizer(line):
    # This outputs a dense version of an unordered tensor with one-hot vectors 
    # at each position. This is to reduce memory consumption, though will have
    # tradeoffs during extraction.
    features = []
    
    linesplit = line.split()
    #start by limiting inputsize
    for i in range(len(linesplit)):
    #for i in range(min(20,len(linesplit))):
        word = linesplit[i]
        if word in worddict:
            features.append(worddict[word])
        else:
            features.append(worddict["<UNK>"])

    #for j in range(max(0,20-len(linesplit))):
    #    features.append(worddict["<EOS>"])
    
    return features
    

def getMinibatch(featureList):
    # This converts the dense version into a sparse *unordered* tensor with
    # counts for each word's frequency in the sample.
    minibatch = []
    length = len(worddict.keys())

    for features in featureList:
        featuresample = [0] * length
        #featuresample = []
        for i in range(len(features)):
            #featureword = [0] * length
            #featureword[features[i]] = 1
            #featuresample.extend(featureword)
            featuresample[features[i]] += 1
            
        minibatch.append(featuresample)

    minibatchtens = torch.tensor(minibatch,dtype=torch.float)
    return minibatchtens

def classify(trainpath, valpath, testpath):
    print("Running...")
    
    #these are all tensors (except size)
    trainfeatures, trainlabel, trainsize = getLabelLine(trainpath,"train")
    valfeatures, vallabel, valsize = getLabelLine(valpath,"val")
    testfeatures, testlabel, testsize = getLabelLine(testpath,"test")

    print("Got data")

    #starting with a fixed input size
    #inputsize = 1379610

    #input is size of vocab
    inputsize = len(worddict.keys())
    hiddensize = 25
    outputsize = len(labeldict)
    
    net = RNNet(inputsize,hiddensize,outputsize)

    print("Built net.")
    
    max_iterations = 4

    learningrate = .05
    moment = 1

    #cross-entropy loss
    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learningrate, momentum=moment)

    for i in range(max_iterations):
        minibatchsize = 300

        #hidint = [[0] * hiddensize] * minibatchsize
        #hidden = torch.tensor(hidint,dtype=torch.float)

        for j in range(int(trainsize / minibatchsize) + 1):
        #for j in range(10):
            start = j * minibatchsize
            end = min((j + 1) * minibatchsize,trainsize)

            print("Epoch " + str(i) + " batch " + str(j))
            
            minibatchfeatures = getMinibatch(trainfeatures[start:end])
            minibatchlabels = torch.tensor(trainlabel[start:end],dtype=torch.long)
            minibatchlabels = minibatchlabels.view(-1)

            optimizer.zero_grad()

            hidint = [[0] * hiddensize] * (end - start)
            hidden = torch.tensor(hidint,dtype=torch.float)
            
            outputs, aux = net(minibatchfeatures,hidden)
            loss = criterion(outputs, minibatchlabels)
            loss.backward()
            optimizer.step()

        # clear memory space before classifying
        minibatchfeatures = []
        minibatchlabels = []
        hidden = []
        outputs = []
        
        test_classifier(net,valfeatures,vallabel,i,"val")
        test_classifier(net,testfeatures,testlabel,i,"test")

def test_classifier(net,inputfeatures,inputlabel,iteration,vers):
    labellist = list(labeldict.keys())
    
    outputpath = "output" + vers + ".txt"
    outputfile = open(outputpath,"w+")
    
    inputsize = len(inputfeatures)
    inputlabeltens = torch.tensor(inputlabel,dtype=torch.float)
    #inputlabeltens = inputlabeltens.view(-1)
    hiddensize = 25
    correct = 0

    #hidint = [[0] * hiddensize] * (end - start)
    #hidden = torch.tensor(hidint,dtype=torch.float)

    output = torch.tensor([])

    for j in range(int(inputsize / 100) + 1):
        start = j * 100
        end = min((j + 1) * 100,inputsize)

        hidint = [[0] * hiddensize] * (end - start)
        hidden = torch.tensor(hidint,dtype=torch.float)
            
        inputbatch = getMinibatch(inputfeatures[start:end])
        #labels = inputlabel[start:end]

        outputchunk, aux = net(inputbatch,hidden)
        output = torch.cat((output,outputchunk))

    for i in range(len(inputfeatures)):
        outputlist = output[i].tolist()
        outputmax = outputlist.index(max(outputlist))
        outputkey = labellist[outputmax]
        outputfile.write(str(outputkey) + "\n")
        if outputmax == inputlabeltens[i]:
            correct += 1
        else:
            #print(inputfeatures[i])
            print("==> Guess: " + str(outputmax) + " Actual: " + str(inputlabel[i]))

    accuracy = round(float(correct)/len(inputfeatures),4)
    print("Accuracy of " + vers + str(iteration) + ": " + str(accuracy))

    outputfile.close()

if __name__ == '__main__':
    trainpath = sys.argv[1]
    valpath = sys.argv[2]
    testpath = sys.argv[3]

    classify(trainpath,valpath,testpath)
