from flask import Flask, jsonify, request , render_template
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import os
import cleanImage
import imageToText
import genQA
import bucket
from google.cloud import datastore
import datetime
  
# creating a Flask app 
app = Flask(__name__)
CORS(app, support_credentials=True) 
  
@app.route('/', methods = ['GET', 'POST']) 
@cross_origin(supports_credentials=True)
def status():
    return render_template('index.html')

@app.route('/scan', methods = ['POST']) 
@cross_origin(supports_credentials=True)
def scan(): 
    file=request.files['img']
    filename = secure_filename(file.filename)
    # file.save(filename)
    bucket.gcs_upload_image(filename)
    text=imageToText.detect_text(filename)
    return jsonify({'data': text}) 

@app.route('/genAI', methods = ['POST']) 
@cross_origin(supports_credentials=True)
def genAI(): 
    data=request.get_json()
    print(data)
    text=data['text']
    response=genQA.genQA(text)
    return jsonify({'data': response}) 

@app.route('/processImage', methods = ['POST']) 
@cross_origin(supports_credentials=True)
def processImage(): 
    file=request.files.get('img')
    sanitized_filename = file.filename.replace(" ", "_")
    tmp_file = f'/tmp/{sanitized_filename}'
    file.save(tmp_file)
    bucket.gcs_upload_image(tmp_file)

    cleanedFile = cleanImage.cleanImage(tmp_file)

    # return bucket.download_from_gcs(cleanedFile)
    cleaned_image_url = bucket.get_gcs_url(cleanedFile)
    
    text=imageToText.detect_text(tmp_file)
    response=genQA.genQA(text)

    return jsonify({
        "cleaned_image_url": cleaned_image_url,
        "qa_response": response
    })

@app.route('/checkAnswer', methods = ['POST']) 
@cross_origin(supports_credentials=True)
def checkAnswer():
    data=request.get_json()
    print(data)
    userAnswer=data['userAnswer']
    correctAnswer=data['correctAnswer']
    question=data['question']
    response=genQA.checkAnswer(userAnswer, correctAnswer, question)
    return jsonify({'data': response})

@app.route('/uploadFeedback', methods = ['POST']) 
@cross_origin(supports_credentials=True)
def uploadFeedback():
    data=request.get_json()
    datastore_client = datastore.Client()
    entity = datastore.Entity(key=datastore_client.key("feedback", data['user']))
    time = datetime.datetime.now()
    entity["{:%B %d, %Y}".format(time)] = data["feedback"]
    datastore_client.put(entity)
    return jsonify(data["feedback"])
 
@app.route('/getImage/<filename>', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_image(filename):
    print("hello")
    return bucket.download_from_gcs(filename)  # Call function from bucket.py


# driver function 
if __name__ == '__main__': 
  
    app.run(debug = True)