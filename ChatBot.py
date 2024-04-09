import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import random
import tensorflow
import json
import pickle
import autocorrect
from autocorrect import spell
import speech_recognition as sr
import pyttsx3
from DbConnect import PnrStatus


with open("intents.json") as file:
    data = json.load(file)
try:
    with open("data.pickle" , "rb") as f:
        words , labels , training , output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
                       
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x , doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1) 
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle" , "wb") as f:
        pickle.dump((words , labels , training , output),f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape = [None , len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net ,8)
net = tflearn.fully_connected(net,len(output[0]), activation="softmax" , name='my_output')
net = tflearn.regression(net,optimizer="adam")

model = tflearn.DNN(net)
try:                                                       
    model.load("model.tflearn")
except:
    model.fit(training , output , n_epoch=1000 , batch_size=8 , show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(words.lower()) for words in s_words]

    for se in s_words:
        for i , w in enumerate(words):
            if w==se:
                bag[i] = 1
    return numpy.array(bag)

def chat():
    
    print("start taliking with bot!")
    while(True):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source,duration=1)
            print("Say Something")
            audio = r.listen(source)
        try:
            inp = r.recognize_google(audio)
            print("you : "+inp)
        except:
            chat()
        if inp.lower() == "quit":
            break
        result = model.predict([bag_of_words(inp , words)])[0]
        results_index = numpy.argmax(result)
        tag = labels[results_index]

        if result[results_index] > 0.6:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg["responses"]
                    bot_response = random.choice(responses)
                    print("Railway Support : "+bot_response)
                    if(bot_response=="Please provide me your PNR" or bot_response=="Can I know your PNR"):
                        engine = pyttsx3.init()
                        en_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
                        engine.setProperty('voice',en_voice_id)
                        engine.say(bot_response)
                        engine.setProperty('rate',120)
                        engine.setProperty('volume', 0.9)
                        engine.runAndWait() 
                        obj = PnrStatus()
                        obj.getStatus()
                    elif(bot_response=="Please provide me train number"):
                        engine = pyttsx3.init()
                        en_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
                        engine.setProperty('voice',en_voice_id)
                        engine.say(bot_response)
                        engine.setProperty('rate',120)
                        engine.setProperty('volume', 0.9)
                        engine.runAndWait() 
                        obj1 = PnrStatus()
                        obj1.getTrainStatus()
                    else:
                        engine = pyttsx3.init()
                        en_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
                        engine.setProperty('voice',en_voice_id)
                        engine.say(bot_response)
                        engine.setProperty('rate',120)
                        engine.setProperty('volume', 0.9)
                        engine.runAndWait()            
        else:
            engine = pyttsx3.init()
            print("I didn't get that , try again,\ Developer is still working on me !!")
            en_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
            engine.setProperty('voice',en_voice_id)
            engine.say("I didn't get that , try again, Developer is still working on me !!")
            engine.setProperty('rate',120)
            engine.setProperty('volume', 0.9)
            engine.runAndWait()
            
chat()

