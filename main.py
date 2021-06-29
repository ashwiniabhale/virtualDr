from flask import Flask
import sqlite3
from flask import Flask, render_template, request
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
#from werkzeug import secure_filename
#  Image Detection
import numpy as np
import os
import tensorflow as tf
from PIL import Image
import pickle
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import warnings
warnings.filterwarnings("ignore")
####################################
# Import dependencies
#from keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
import h5py
from PIL import Image
import PIL
from vb100_utils import *
################################
app = Flask(__name__, static_url_path='',
            static_folder='static',
            template_folder='templates')

@app.route('/')  # @ decorator
def main_page():
    return render_template('index.html')

@app.route('/apoint', methods = ['GET'])
def appointment():
    return render_template('appointment.html')

@app.route('/addrec',methods = ['POST'])
def addrec():
   if request.method == 'POST':
       with sqlite3.connect('database.db') as con:
           cur = con.cursor()
           try:
               name = request.form['name']
               email = request.form['email']
               date = request.form['date']
               department = request.form['department']
               mobile = request.form['phone']
               info = request.form['message']
               mail_content = 'Hello ' + str(name) + ', \n\nYour appointment is book on date (YYYY-MM-DD) : ' + date + '\n\nThank You.'
               message = MIMEMultipart()
               message['From'] = 'pandharkarneha19@gmail.com'
               message['To'] = email
               message['Subject'] = 'Appointment Booking Confirmation.'
               sender_address = 'pandharkarneha19@gmail.com'
               sender_pass = 'Neha@1999'
               receiver_address = email
               message.attach(MIMEText(mail_content, 'plain'))
               # Create SMTP session for sending the mail
               session = smtplib.SMTP('smtp.gmail.com', 587)  # use gmail with port
               session.starttls()  # enable security
               session.login(sender_address, sender_pass)  # login with mail_id and password
               text = message.as_string()
               session.sendmail(sender_address, receiver_address, text)
               session.quit()
               cur.execute("INSERT INTO appointment (name, email, date, department, mobile, info ) "
                          "VALUES(?, ?, ?, ?, ?, ?)",(name,email,date,department,mobile, info) )
               con.commit()
               msg = "Record successfully added"
           except:
               con.rollback()
               msg = "Error In connection"
       con.close()
       #return redirect(url_for('result'))
   return render_template("appointment.html")

@app.route('/imageupload', methods = ['GET'])
def imageupload():
    return render_template('earlytest.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        print("Predict method call")
        model = load_model('AlzheimerDetection.h5')
        print("Load Model")
        # route to any of the labaled malignant images that model hasn't seen before
        # Get the file from post request
        name = request.form['name']
        email = request.form['email']
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img_path = file_path
        print("Take Image")
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)

        # make prediction
        rs = model.predict(img_data)
        print(rs)

        value1 = float(rs[0][0])
        value2 = float(rs[0][1])
        if (value1 < value2):
            predicted_class = 'This image is NOT Alzheimer.'
        else:
            predicted_class = 'Warning! This image IS Alzheimer.'
        print(predicted_class)
        return str(predicted_class)

@app.route('/treatment', methods = ['GET'])
def treatment():
    return render_template('treatment.html')

@app.route('/treatmentstart',methods = ['POST'])
def treatmentstart():
   if request.method == 'POST':
       try:
           department = str(request.form['department']).lower()
           if(department == 'covid'):
               return redirect(url_for('covid'))
           elif(department == 'pneumonia'):
               return redirect(url_for('pneumonia'))
           elif (department == 'brain'):
               return redirect(url_for('brain'))
           elif (department == 'alzheimer'):
               return redirect(url_for('alzheimer'))
           # elif (department == 'heart'):
           #     return redirect(url_for('heart'))
           # elif (department == 'liver'):
           #     return redirect(url_for('liver'))
           # elif (department == 'breast'):
           #     return redirect(url_for('breast'))
           elif (department == 'malaria'):
               return redirect(url_for('malaria'))
           else:
               pass
       except:
           return render_template('index.html')
#---------------------------------------------------1
@app.route('/covid', methods = ['GET'])
def covid():
    return render_template('covidtest.html')
#---------------------------------------------------2
@app.route('/pneumonia', methods = ['GET'])
def pneumonia():
    return render_template('pnemoniatest.html')
#---------------------------------------------------3
@app.route('/brain', methods = ['GET'])
def brain():
    return render_template('braintest.html')
#---------------------------------------------------4
@app.route('/alzheimer', methods = ['GET'])
def alzheimer():
    return render_template('alzheimertest.html')
#---------------------------------------------------5
# @app.route('/heart', methods = ['GET'])
# def heart():
#     return render_template('hearttest.html')
# #---------------------------------------------------6
# @app.route('/liver', methods = ['GET'])
# def liver():
#     return render_template('livertest.html')
# #---------------------------------------------------7
# @app.route('/breast', methods = ['GET'])
# def breast():
#     return render_template('breasttest.html')
# #---------------------------------------------------8
@app.route('/malaria', methods = ['GET'])
def malaria():
    return render_template('maleriatest.html')
#---------------------------------------------------1
@app.route('/covidtest', methods = ['POST'])
def covidtest():
    if request.method == 'POST':
        json_file = open('model_adam_covid.json')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # Get weights into the model
        loaded_model.load_weights('model_100_eopchs_adam_covid.h5')
        loaded_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img_path = file_path
        print("Take Image")
        IMG = tf.keras.preprocessing.image.load_img(img_path)
        #IMG = Image.open('covid.jpeg')
        print(type(IMG))
        IMG = IMG.resize((342, 257))
        IMG = np.array(IMG)
        print('po array = {}'.format(IMG.shape))
        IMG = np.true_divide(IMG, 255)
        IMG = IMG.reshape(3, 342, 257, 1)
        print(type(IMG), IMG.shape)

        predictions = loaded_model.predict(IMG)

        print(loaded_model)
        predictions_c = loaded_model.predict_classes(IMG)
        print(predictions, predictions_c)
        classes = {'TRAIN': ['BACTERIA', 'NORMAL', 'VIRUS'],
                   'VALIDATION': ['BACTERIA', 'NORMAL'],
                   'TEST': ['BACTERIA', 'NORMAL', 'VIRUS']}

        predicted_class = classes['TRAIN'][predictions_c[0]]
        result = 'We think that is {}.'.format(predicted_class.lower())
        #return redirect(url_for('.result', prediction_text = result))
        return result
#---------------------------------------------------2
@app.route('/pneumoniatest', methods = ['POST'])
def pneumoniatest():
    if request.method == 'POST':
        json_file = open('model_adam_pnemonia.json')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # Get weights into the model
        loaded_model.load_weights('model_100_eopchs_adam_pnemonia.h5')
        loaded_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img_path = file_path
        print("Take Image")
        IMG = tf.keras.preprocessing.image.load_img(img_path)
        # IMG = Image.open('covid.jpeg')
        print(type(IMG))
        IMG = IMG.resize((342, 257))
        IMG = np.array(IMG)
        print('po array = {}'.format(IMG.shape))
        IMG = np.true_divide(IMG, 255)
        IMG = IMG.reshape(3, 342, 257, 1)
        print(type(IMG), IMG.shape)

        predictions = loaded_model.predict(IMG)

        print(loaded_model)
        predictions_c = loaded_model.predict_classes(IMG)
        print(predictions, predictions_c)
        classes = {'TRAIN': ['Normal', 'Mid-Infected', 'High-Infected'],
                   'VALIDATION': ['Normal', 'Mid-Infected'],
                   'TEST': ['Normal', 'Mid-Infected', 'High-Infected']}

        predicted_class = classes['TRAIN'][predictions_c[0]]
        result = 'We think that is {}.'.format(predicted_class.lower())
    return result
#---------------------------------------------------3
@app.route('/braintest', methods = ['POST'])
def braintest():
    if request.method == 'POST':
        print("Predict method call")
        model = load_model('AlzheimerDetection.h5')
        print("Load Model")
        # route to any of the labaled malignant images that model hasn't seen before
        # Get the file from post request
        name = request.form['name']
        email = request.form['email']
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img_path = file_path
        print("Take Image")
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)

        # make prediction
        rs = model.predict(img_data)
        print(rs)

        value1 = float(rs[0][0])
        value2 = float(rs[0][1])
        if (value1 < value2):
            predicted_class = 'This image is NOT brain tumor.'
        else:
            predicted_class = 'Warning! This image Is brain tumor.'
    result = 'We think that is {}.'.format(predicted_class)
    return result
#---------------------------------------------------4
@app.route('/alzheimertest', methods = ['POST'])
def alzheimertest():
    if request.method == 'POST':
        print("Predict method call")
        model = load_model('AlzheimerDetection.h5')
        print("Load Model")
        # route to any of the labaled malignant images that model hasn't seen before
        # Get the file from post request
        name = request.form['name']
        email = request.form['email']
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img_path = file_path
        print("Take Image")
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)

        # make prediction
        rs = model.predict(img_data)
        print(rs)

        value1 = float(rs[0][0])
        value2 = float(rs[0][1])
        if (value1 < value2):
            predicted_class = 'This image is NOT Alzheimer.'
        else:
            predicted_class = 'Warning! This image IS Alzheimer.'
        result = 'We think that is {}.'.format(predicted_class)
        return result
# #---------------------------------------------------5
# @app.route('/hearttest', methods = ['POST'])
# def hearttest():
#     if request.method == 'POST':
#         model_heartdisease = pickle.load(open('heartdisease.pkl', 'rb'))
#         Age = int(request.form['age'])
#         Gender = int(request.form['sex'])
#         ChestPain = int(request.form['cp'])
#         BloodPressure = int(request.form['trestbps'])
#         ElectrocardiographicResults = int(request.form['restecg'])
#         MaxHeartRate = int(request.form['thalach'])
#         ExerciseInducedAngina = int(request.form['exang'])
#         STdepression = float(request.form['oldpeak'])
#         ExercisePeakSlope = int(request.form['slope'])
#         MajorVesselsNo = int(request.form['ca'])
#         Thalassemia = int(request.form['thal'])
#         prediction = model_heartdisease.predict([[Age, Gender, ChestPain, BloodPressure, ElectrocardiographicResults,
#                                                   MaxHeartRate, ExerciseInducedAngina, STdepression, ExercisePeakSlope,
#                                                   MajorVesselsNo, Thalassemia]])
#         if prediction == 1:
#             predicted_class="Oops! You seem to have a Heart Disease."
#             result = 'We think that is {}.'.format(predicted_class)
#             return result
#         else:
#             predicted_class="Great! You don't have any Heart Disease."
#             result = 'We think that is {}.'.format(predicted_class)
#             return result
#     else:
#         return render_template('index.html')
# #---------------------------------------------------6
# @app.route('/livertest', methods = ['POST'])
# def livertest():
#     if request.method == 'POST':
#         model_liverdisease = pickle.load(open('liverdisease.pkl', 'rb'))
#         Age = int(request.form['Age'])
#         Gender_D = request.form['Gender']
#         if(Gender_D == "Male"):
#             Gender = 1
#         else:
#             Gender = 0
#         Total_Bilirubin = float(request.form['Total_Bilirubin'])
#         Direct_Bilirubin = float(request.form['Direct_Bilirubin'])
#         Alkaline_Phosphotase = int(request.form['Alkaline_Phosphotase'])
#         Alamine_Aminotransferase = int(request.form['Alamine_Aminotransferase'])
#         Aspartate_Aminotransferase = int(request.form['Aspartate_Aminotransferase'])
#         Total_Protiens = float(request.form['Total_Protiens'])
#         Albumin = float(request.form['Albumin'])
#         Albumin_and_Globulin_Ratio = float(request.form['Albumin_and_Globulin_Ratio'])
#         prediction = model_liverdisease.predict([[Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase,
#                                                   Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens,
#                                                   Albumin, Albumin_and_Globulin_Ratio]])
#         if prediction == 1:
#             predicted_class = "Oops! You seem to have a Liver Disease."
#             result = 'We think that is {}.'.format(predicted_class)
#             return result
#         else:
#             predicted_class = "Great! You don't have any Liver Disease."
#             result = 'We think that is {}.'.format(predicted_class)
#             return result
# #---------------------------------------------------7
# @app.route('/breasttest', methods = ['POST'])
# def breasttest():
#     if request.method == 'POST':
#         model_cancer = pickle.load(open('breastcancer.pkl', 'rb'))
#         texture_mean = float(request.form['texture_mean'])
#         perimeter_mean = float(request.form['perimeter_mean'])
#         smoothness_mean = float(request.form['smoothness_mean'])
#         compactness_mean = float(request.form['compactness_mean'])
#         concavity_mean = float(request.form['concavity_mean'])
#         concave_points_mean = float(request.form['concave_points_mean'])
#         symmetry_mean = float(request.form['symmetry_mean'])
#         radius_se = float(request.form['radius_se'])
#         compactness_se = float(request.form['compactness_se'])
#         concavity_se = float(request.form['concavity_se'])
#         concave_points_se = float(request.form['concave_points_se'])
#         texture_worst = float(request.form['texture_worst'])
#         smoothness_worst = float(request.form['smoothness_worst'])
#         compactness_worst = float(request.form['compactness_worst'])
#         concavity_worst = float(request.form['concavity_worst'])
#         concave_points_worst = float(request.form['concave_points_worst'])
#         symmetry_worst = float(request.form['symmetry_worst'])
#         fractal_dimension_worst = float(request.form['fractal_dimension_worst'])
#         prediction = model_cancer.predict([[texture_mean, perimeter_mean, smoothness_mean, compactness_mean,
#                                             concavity_mean, concave_points_mean, symmetry_mean, radius_se,
#                                             compactness_se, concavity_se, concave_points_se, texture_worst,
#                                             smoothness_worst, compactness_worst, concavity_worst,
#                                             concave_points_worst, symmetry_worst, fractal_dimension_worst]])
#         if prediction == 1:
#             predicted_class = "Oops! The tumor is malignant."
#             result = 'We think that is {}.'.format(predicted_class)
#             return result
#         else:
#             predicted_class = "Great! You don't have any Liver Disease."
#             result = 'We think that is {}.'.format(predicted_class)
#             return result
#---------------------------------------------------8
# Image Preprocessing
def malaria_predict(img_path):
    model_malaria = load_model('malariadisease.h5')
    img = image.load_img(img_path, target_size=(30, 30, 3))
    x=image.img_to_array(img)
    x=x/255
    x=np.expand_dims(x, axis=0)
    preds = model_malaria.predict(x)
    return preds

@app.route('/malariatest', methods = ['POST'])
def malariatest():
    if request.method == 'POST':
        f = request.files["file"]
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        prediction = malaria_predict(file_path)
        if prediction[0][0] >= 0.5:
           prediction_text="Oops! The cell image indicates the presence of Malaria."
           return prediction_text
        else:
            prediction_text="Great! You don't have Malaria."
            return prediction_text
#---------------------------------------------------
if __name__ == '__main__':
   app.run(debug = True, host = '0.0.0.0')