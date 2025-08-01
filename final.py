#import pyswarms as ps
#from SwarmPackagePy import testFunctions as tf #load pso particle swarm package
import pandas as pd
import numpy as np
import os
from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory
import base64
import io
import cv2
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from keras.applications import ResNet50
from sklearn.metrics import accuracy_score
from keras.layers import AveragePooling2D
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt #use to visualize dataset values

app = Flask(__name__)
app.secret_key = 'welcome'

#define and load class labels found in dataset
path = "Dataset"
labels = []
X = []
Y = []
for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name.strip())
print("Medicinal Plants Class Labels : "+str(labels)) 

#define function to get class label of given image
def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

    #load dataset image and process them
if os.path.exists("model/X.txt.npy"):
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
else: #if images not process then read and process image pixels
    X.clear()
    Y.clear()
    for root, dirs, directory in os.walk(path):#connect to dataset folder
        for j in range(len(directory)):#loop all images from dataset folder
            name = os.path.basename(root)
            if 'Thumbs.db' not in directory[j]:
                img = cv2.imread(root+"/"+directory[j])#read images
                img = cv2.resize(img, (32, 32))#resize image
                X.append(img) #add image pixels to X array
                label = getLabel(name)#get image label id
                Y.append(label)#add image label                
    X = np.asarray(X)#convert array as numpy array
    Y = np.asarray(Y)
    np.save('model/X.txt',X)#save process images and labels
    np.save('model/Y.txt',Y)
print("Dataset images loaded")
print("Total images found in dataset : "+str(X.shape[0]))
print()

X = X.astype('float32')
X = X/255 #normalized pixel values between 0 and 1
indices = np.arange(X.shape[0])
np.random.shuffle(indices)#shuffle all images
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print("Dataset Image Processing & Normalization Completed")
print("80% images used to train algorithms : "+str(X_train.shape[0]))
print("20% image used to test algorithms : "+str(X_test.shape[0]))

accuracy = []
precision = []
recall = []
fscore = []

resnet_model = ResNet50(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
for layer in resnet_model.layers:
    layer.trainable = False
resnet_model = Sequential()
resnet_model.add(Convolution2D(32, (3 , 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
resnet_model.add(MaxPooling2D(pool_size = (2, 2)))
resnet_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
resnet_model.add(MaxPooling2D(pool_size = (2, 2)))
resnet_model.add(Flatten())
resnet_model.add(Dense(units = 256, activation = 'relu'))
resnet_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
resnet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/resnet_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/resnet_weights.hdf5', verbose = 1, save_best_only = True)
    hist = resnet_model.fit(X_train, y_train, batch_size = 32, epochs = 30, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/resnet_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    resnet_model.load_weights("model/resnet_weights.hdf5")
#extracting features from trained Resnet50 model
print("Number of Features exists in each image before applying ResNet50 : "+str(X.shape[1] * X.shape[2] * X.shape[3]))
#creating Resnet features extraction model
cascade_model = Model(inputs = resnet_model.inputs, outputs = resnet_model.layers[-2].output)#getting ANN model layers
X = cascade_model.predict(X)#extracting hybrid features from TEXTCNN
Y = np.argmax(Y, axis=1)
print("Total Features Extracted from each image after applying ResNet50 : "+str(X.shape[1]))

#Defining function to apply PSO algorithm on ResNet50 features 
classifier = RandomForestClassifier()
print("ResNet Features before applying PSO : "+str(X.shape[1]))
def f_per_particle(m, alpha):
    global X, Y, classifier
    total_features = X.shape[1]
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]
    classifier.fit(X_subset, Y)
    P = (classifier.predict(X_subset) == Y).mean()
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
    return j

def f(x, alpha=0.88):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)

def runPSO():
    global X, Y, pso
    if os.path.exists("model/pso.npy"):
        pso = np.load("model/pso.npy")
    else:
        options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 5, 'p':2}
        dimensions = X.shape[1] # dimensions should be the number of features
        optimizer = ps.discrete.BinaryPSO(n_particles=10, dimensions=dimensions, options=options) #CREATING PSO OBJECTS 
        cost, pso = optimizer.optimize(f, iters=35)#OPTIMIZING FEATURES
        np.save("model/pso", pso)
    return pso
pso = runPSO()
X = X[:,pso==1]
print("Resnet Features after applying PSO : "+str(X.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data
print("ResNet50 & PSO Features Train & Test Split Details")
print("80% images used to train algorithms : "+str(X_train.shape[0]))
print("20% image used to test algorithms : "+str(X_test.shape[0]))

model_path = 'model/extension_svm_model.pkl'

# Check if the model already exists and load it if available
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        extension_svm = pickle.load(f)
    print("Loaded pre-existing Extension Tuned SVM model.")
else:
    # Initialize a new SVC with specific kernel, C, and gamma parameters
    extension_svm = svm.SVC(kernel='rbf', C=150, gamma='auto')
    print("Training new Extension Tuned SVM model.")
    
    # Train the SVM model on the training data
    extension_svm.fit(X_train, y_train)
    
    # Save the trained model to file
    with open(model_path, 'wb') as f:
        pickle.dump(extension_svm, f)
    print("Saved trained Extension Tuned SVM model to file.")

# Perform prediction on test data
predict = extension_svm.predict(X_test)

def classifyPlantType(image_path):
    image = cv2.imread(image_path)#read test image
    img = cv2.resize(image, (32,32))#resize image
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)#convert image as 4 dimension
    img = np.asarray(im2arr)
    img = img.astype('float32')#convert image features as float
    img = img/255 #normalized image
    resnet_features = cascade_model.predict(img)#extract Resnet features from trained model
    resnet_features = resnet_features[:,pso==1]#apply PSO to select features
    predict = extension_svm.predict(resnet_features)#apply extension Tuned SVM on PSO features to predict medicinal plant type
    predict = predict[0]
    img = cv2.imread(image_path)
    img = cv2.resize(img, (300,200))#display image with predicted output
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.putText(img, 'Predicted As : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    plt.imshow(img)

def getModel():
    resnet_model = ResNet50(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
    for layer in resnet_model.layers:
        layer.trainable = False
    resnet_model = Sequential()
    resnet_model.add(Convolution2D(32, (3 , 3), input_shape = (32, 32, 3), activation = 'relu'))
    resnet_model.add(MaxPooling2D(pool_size = (2, 2)))
    resnet_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
    resnet_model.add(MaxPooling2D(pool_size = (2, 2)))
    resnet_model.add(Flatten())
    resnet_model.add(Dense(units = 256, activation = 'relu'))
    resnet_model.add(Dense(units = 7, activation = 'softmax'))
    resnet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    resnet_model.load_weights("model/resnet_weights.hdf5")
    return resnet_model

@app.route('/Predict', methods=['GET', 'POST'])
def predictView():
    return render_template('Predict.html', msg='')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', msg='')

@app.route('/index', methods=['GET', 'POST'])
def index1():
    return render_template('index.html', msg='')

@app.route('/AdminLogin', methods=['GET', 'POST'])
def AdminLogin():
    return render_template('AdminLogin.html', msg='')

@app.route('/AdminLoginAction', methods=['GET', 'POST'])
def AdminLoginAction():
    if request.method == 'POST' and 't1' in request.form and 't2' in request.form:
        user = request.form['t1']
        password = request.form['t2']
        if user == "admin" and password == "admin":
            return render_template('AdminScreen.html', msg="Welcome "+user)
        else:
            return render_template('AdminLogin.html', msg="Invalid login details")

@app.route('/Logout')
def Logout():
    return render_template('index.html', msg='')

@app.route('/PredictAction', methods=['GET', 'POST'])
def PredictAction():   
    if request.method == 'POST':
        file = request.files['t1']
        img_bytes = file.read()
        if os.path.exists("static/test.jpg"):
            os.remove("static/test.jpg")
        with open('static/test.jpg', mode="wb") as jpg:
            jpg.write(img_bytes)
        jpg.close()
        extension_model = getModel()
        image = cv2.imread('static/test.jpg')#read test image   
        img = cv2.resize(image, (32, 32))#resize image
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,32,32,3)#convert image as 4 dimension
        img = np.asarray(im2arr)
        img = img.astype('float32')#convert image features as float
        img = img/255 #normalized image
        predict = extension_model.predict(img)#now predict crack
        predict = np.argmax(predict)
        img = cv2.imread('static/test.jpg')
        img = cv2.resize(img, (300,200))#display image with predicted output
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output = 'Predicted As : '+labels[predict]
        cv2.putText(img, 'Predicted As : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
        plt.imshow(img)        
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode() 
        return render_template('AdminScreen.html', msg=output, img = img_b64)

if __name__ == '__main__':
    app.run()
