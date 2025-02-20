from flask import Flask, jsonify, request , render_template
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import os
import cleanImage
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
    bucket.gcs_upload_image(tmp_file)

    cleanedFile = cleanImage.cleanImage(tmp_file)

    # return bucket.download_from_gcs(cleanedFile)
    cleaned_image_url = bucket.get_gcs_url(cleanedFile)
    
    text=imageToText.detect_text(tmp_file)
    response=genQA.genQA(text)
    # return jsonify({'data': response})

    return jsonify({
        "cleaned_image_url": cleaned_image_url,
        "qa_response": response
    })
 
@app.route('/getImage/<filename>', methods=['GET'])
@cross_origin(supports_credentials=True)
def get_image(filename):
    print("hello")
    return bucket.download_from_gcs(filename)  # Call function from bucket.py


# driver function 
if __name__ == '__main__': 
  
    app.run(debug = True) 