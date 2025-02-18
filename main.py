from flask import Flask, jsonify, request , render_template
import handwritingToText
import imageToText
import genQA
  
# creating a Flask app 
app = Flask(__name__) 
  
@app.route('/', methods = ['GET', 'POST']) 
def status():
    return render_template('index.html')
    # return "Server is up"

@app.route('/scan', methods = ['GET', 'POST']) 
def scan(): 
    if(request.method == 'GET'): 
        #cleanedImage = cleanImage.cleanImage('test.jpeg')
        #text=handwritingToText.detectDocument('handwriting.jpeg')
        text=imageToText.detect_text('scan.png')
        return jsonify({'data': text}) 

@app.route('/genAI', methods = ['POST']) 
def genAI(): 
    if(request.method == 'GET'): 
        #cleanedImage = cleanImage.cleanImage('test.jpeg')
        #text=handwritingToText.detectDocument('handwriting.jpeg')
        data=request.get_json()
        print(data)
        text=data['text']
        response=genQA.genQA(text)
        return jsonify({'data': response}) 

@app.route('/complete', methods = ['GET', 'POST']) 
def complete(): 
    if(request.method == 'GET'): 
        #cleanedImage = cleanImage.cleanImage('test.jpeg')
        #text=handwritingToText.detectDocument('handwriting.jpeg')
        text=imageToText.detect_text('scan.png')
        response=genQA.genQA(text)
        return jsonify({'data': response}) 
  
  
# A simple function to calculate the square of a number 
# the number to be squared is sent in the URL when we use GET 
# on the terminal type: curl http://127.0.0.1:5000 / home / 10 
# this returns 100 (square of 10) 
@app.route('/home/<int:num>', methods = ['GET']) 
def disp(num): 
  
    return jsonify({'data': num**2}) 
  
  
# driver function 
if __name__ == '__main__': 
  
    app.run(debug = True) 