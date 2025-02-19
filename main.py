from flask import Flask, jsonify, request , render_template
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import os
import handwritingToText
import imageToText
import genQA
import bucket
  
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
    tmp_file = f'/tmp/{file.filename}'
    file.save(tmp_file)
    # filename = secure_filename(file.filename)
    bucket.gcs_upload_image(tmp_file)
    text=imageToText.detect_text(tmp_file)
    response=genQA.genQA(text)
    return jsonify({'data': response})
 
# driver function 
if __name__ == '__main__': 
  
    app.run(debug = True) 