from builtins import range

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import pickle
import random
import cv2

def uloha1():
    fruits=os.listdir('fruits-360/Training')
    for fruitname in fruits:
        fruit=os.listdir('fruits-360/Training/'+fruitname)
        for x in range(4):
            img=pltimg.imread('fruits-360/Training/'+fruitname+'/'+fruit[x])
            plt.imshow(img)
            plt.show()
    plt.close("all")

def uloha1test():
    fruits = os.listdir('fruits-360/test-multiple-fruits')
    for fruitname in fruits:
        fruit = os.listdir('fruits-360/test-multiple-fruits/' + fruitname)
        for x in range(4):
            img = pltimg.imread('fruits-360/test-multiple-fruits/' + fruitname + '/' + fruit[x])
            plt.imshow(img)
            plt.show()
    plt.close("all")

def uloha2():
    fruits = os.listdir('fruits-360/Training')
    for fruitName in fruits:
        fruit = os.listdir('fruits-360/Training/' + fruitName)
        fruitPhotos=[]
        for fruitPhoto in fruit:
            img=plt.imread('fruits-360/Training/'+fruitName+'/'+fruitPhoto)
            img=img/510
            fruitPhotos.append(img)
        pickling_on=open(fruitName+".pickle","wb")
        pickle.dump(fruitPhotos,pickling_on)
        pickling_on.close()

def showImage(image,title):
    plt.imshow(image)
    plt.title(title)
    plt.show()

def showImageCV(image,title):
    cv2.namedWindow(title)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    cv2.imshow(title, image)
    cv2.waitKey(0)

def unpickleFile(filename):
    pickle_off=open(filename,'rb')
    return pickle.load(pickle_off)

def loadData():
    print('TODO')

def loadFruitNames():
    fruits=os.listdir('fruits-360/Training')
    return fruits

def normalizeImage(image):
    norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_image

def uloha2v2():
    fruits = os.listdir('fruits-360/Training')
    for fruitName in fruits:
        fruit = os.listdir('fruits-360/Training/' + fruitName)
        fruitPhotos = []
        for fruitPhoto in fruit:
            img = plt.imread('fruits-360/Training/' + fruitName + '/' + fruitPhoto)
            img = normalizeImage(img)
            fruitPhotos.append(img)
        pickling_on = open(fruitName + ".pickle", "wb")
        pickle.dump(fruitPhotos, pickling_on)
        pickling_on.close()

def uloha3():
    fruits=loadFruitNames()
    for fruitName in fruits:
        fruit=unpickleFile(fruitName+'.pickle')
        for fruitImageData in fruit:
            showImageCV(fruitImageData,fruitName)

def uloha3v2():
    files=os.listdir("./")
    for file in files:
     if '.pickle' in file:
        fruit=unpickleFile(file)
        for fruitImageData in  fruit:
            showImageCV(fruitImageData,file)

def uloha4():
    fruits = os.listdir('fruits-360/Training')
    for fruit in fruits:
        fruitImages=os.listdir('fruits-360/Training/'+fruit)
        length=len(fruitImages)
        print(fruit+': '+str(length)+' obrazkov')



def uloha5():
    path=os.listdir("./")
    fruitAllPhotos=[];
    pickling_on = open("all.pickle", "wb")
    for x in range(5):
        fruitPhotos=unpickleFile("./"+path[x])
        for y in fruitPhotos:
            fruitAllPhotos.append(y)
    fruitAllPhotos=random.shuffle(fruitAllPhotos)
    pickle.dump(fruitAllPhotos, pickling_on)
    pickling_on.close()

def uloha6():
    print("TODO")


def uloha7():
    print("TODO")

def printAllFileNames():
    files=os.listdir("./")
    for file in files:
        print(file)

def checkPickleFilesLength():
    files=os.listdir("./")
    for file in files:
        if '.pickle' in file:
            photos=unpickleFile(file)
            length=len(photos)
            print("Dlzka file: "+file+" je "+ str(length))

def loadCertainNumberOfPhotosToPickleFiles(number):
    fruits = os.listdir('fruits-360/Training')
    for fruitName in fruits:
        fruit = os.listdir('fruits-360/Training/' + fruitName)
        fruitPhotos = []
        i=0;
        for fruitPhoto in fruit:
            if i<=number:
                img = plt.imread('fruits-360/Training/' + fruitName + '/' + fruitPhoto)
                img = normalizeImage(img)
                fruitPhotos.append(img)
            else:
                break
        pickling_on = open(fruitName + ".pickle", "wb")
        pickle.dump(fruitPhotos, pickling_on)
        pickling_on.close()

if __name__=="__main__":
    printAllFileNames()
    #priemer min odcylka stdev




