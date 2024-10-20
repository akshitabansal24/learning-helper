import nltk
import random
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils
import io
import os
import speech_recognition as sr 
#from gtts import gTTS
#import os 
import winsound
from win32com.client import Dispatch
from google.cloud import vision
#from google.cloud.vision import types
from google.cloud.vision_v1 import types
from google.cloud.bigquery.client import Client
from fuzzywuzzy import fuzz

class QA:
    def __init__(self,question,actual_answer,question_index):
        self.question=question
        self.actual_answer=actual_answer
        self.accuracy=0
        self.question_index=question_index
        self.response_answer=""
'''s="""Assignment I
Q1: Find the collocations in text5
Q2: Define a variable my_sent to be a list of words. Convert my_sent into string and then split it
as list of words.
Q3: Find the index of the word sunset in text9.
Q4: ompute the vocabulary of the sentences sent1 ... sent8
Q5: What is the difference between the following two lines:
>>> sorted(set([w.lower() for w in text1]))
>>> sorted([w.lower() for w in set(text1)])
Q6: Write the slice expression that extracts the last two words of text2
Q7: Find all the four-letter words in the Chat Corpus (text5). With the help of a frequency
distribution (FreqDist), show these words in decreasing order of frequency
Q8: Use a combination of for and if statements to loop over the words of the movie script for
Monty Python and the Holy Grail (text6) and print all the uppercase words
Q9: Write expressions for finding all words in text6 that meet the following conditions.
a. Ending in ize
b. Containing the letter z
c. Containing the sequence of letters pt
d. All lowercase letters except for an initial capital (i.e., titlecase)
Q10: Define sent to be the list of words ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']. Now
write code to perform the following tasks:
a. Print all words beginning with sh.
b. Print all words longer than four characters
Q11: What does the following Python code do? sum([len(w) for w in text1]) Can you use it to
work out the average word length of a text?
Q12: Define a function called vocab_size(text) that has a single parameter for the text, and which
returns the vocabulary size of the text.
Q13: Define a function percent(word, text) that calculates how often a given word occurs in a
text and expresses the result as a percentage."""
#list1=s.split('Q')
s1="""Assignment I
ANS1: Find the collocations in text5
ANS2: Define a variable my_sent to be a list of words. Convert my_sent into string and then split it
as list of words.
ANS3: Find the index of the word sunset in text9.
ANS4: ompute the vocabulary of the sentences sent1 ... sent8
ANS5: What is the difference between the following two lines:
>>> sorted(set([w.lower() for w in text1]))
>>> sorted([w.lower() for w in set(text1)])
ANS6: Write the slice expression that extracts the last two words of text2
ANS7: Find all the four-letter words in the Chat Corpus (text5). With the help of a frequency
distribution (FreqDist), show these words in decreasing order of frequency
ANS8: Use a combination of for and if statements to loop over the words of the movie script for
Monty Python and the Holy Grail (text6) and print all the uppercase words
ANS9: Write expressions for finding all words in text6 that meet the following conditions.
a. Ending in ize
b. Containing the letter z
c. Containing the sequence of letters pt
d. All lowercase letters except for an initial capital (i.e., titlecase)
ANS10: Define sent to be the list of words ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']. Now
write code to perform the following tasks:
a. Print all words beginning with sh.
b. Print all words longer than four characters
ANS11: Can you use it to work out the average word length of a text?
ANS12: Define a function called vocabulary that has a single parameter for the text and
returns the vocabulary size of the text.
ANS13: Define a function percent(word, text) that calculates how often a given word occurs in a
text and expresses the result as a percentage.
"""
list2=s1.split("ANS")'''
#print("".join(list1[12][4:].splitlines()[0:len(list1[12][4:].splitlines())]))
#print(" ".join(list1[11][list1[11].find(":")+2:].splitlines()[0:len(list1[11][list1[11].find(":")+2:].splitlines())]))


my_objects = []
less_accuracy_question_index=[]
#print(len(list1)-1)
#############################
'''for i in range(len(list1)-1):
    my_objects.append(QA(" ".join(list1[i+1][list1[i+1].find(":")+2:].splitlines()[0:len(list1[i+1][list1[i+1].find(":")+2:].splitlines())])," ".join(list2[i+1][list2[i+1].find(":")+2:].splitlines()[0:len(list2[i+1][list2[i+1].find(":")+2:].splitlines())]),i))'''
#############################
'''my_objects[3].accuracy=40
my_objects[6].accuracy=30
my_objects[1].accuracy=10
my_objects[2].accuracy=60
for i in my_objects :
    if (i.accuracy<20):
        less_accuracy_question_index.append(i)
while less_accuracy_question_index:
    i=random.choice(less_accuracy_question_index)
    if (i.accuracy>20):
        less_accuracy_question_index.remove(i)
    else:
        i.accuracy=random.randrange(20, 40, 3)
print(less_accuracy_question_index)
for i in my_objects :
    print(i.question)'''
#print(my_objects[12].question)

speak = Dispatch("SAPI.SpVoice")
mic_name = "USB Device 0x46d:0x825: Audio (hw:1, 0)"
#below 3 lines
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'apikey.json'
bq_client = Client()
client = vision.ImageAnnotatorClient()

def order_points(pts):

	rect = np.zeros((4, 2), dtype="float32")

	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]


	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect


def transformFourPoints(image, pts):

	rect = order_points(pts)
	(tl, tr, br, bl) = rect


	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))


	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))


	dst = np.array([[0, 0],	[maxWidth - 1, 0],	[maxWidth - 1, maxHeight - 1],	[0, maxHeight - 1]], dtype="float32")


	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped


'''ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image to be scanned")
args = vars(ap.parse_args())'''
image=cv2.imread(r'C:\Users\Akshita\Downloads\The-learning-Helper-master\The-learning-Helper-master\test.jpeg')
#image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 1)
edged = cv2.Canny(gray, 75, 200)


#from transform import transformFourPoints  # Ensure this function is correctly imported

# Assuming 'image' is already defined and loaded
print("STEP 1: Edge Detection")
#cv2.imshow("Image", image)
edged = cv2.Canny(image, 75, 200)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Finding contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

screenCnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:  # Looks for quadrilateral (four points)
        screenCnt = approx
        break

if screenCnt is None:
    print("No contour with 4 points found")
else:
    print("STEP 2: Finding contours of paper")
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Assuming 'orig' and 'ratio' are defined earlier in your code
    warped = transformFourPoints(orig, screenCnt.reshape(4, 2) * ratio)

    # Convert to grayscale and apply adaptive thresholding
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset=10, method="gaussian")
    warped = (warped > T).astype("uint8") * 255

    print("STEP 3: Applying perspective transform")
    cv2.imshow("Original", imutils.resize(orig, height=650))
    cv2.imshow("Scanned", imutils.resize(warped, height=650))
    cv2.imwrite("scan.png", imutils.resize(warped, height=650))
    variable = imutils.resize(warped, height=650)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#below new
file_name = os.path.join(os.path.dirname(__file__),'test2.jpg')


# Loads the image into memory

with open(r'C:\Users\Akshita\Downloads\The-learning-Helper-master\The-learning-Helper-master\test2.jpg', 'rb') as image_file: 
    content = image_file.read()

image = types.Image(content=content)
# Performs label detection on the image file
print("file loaded")
try:
    response = client.text_detection(image=image)
except Exception as e:
    print(f"Error during text detection: {e}")


labels = response.text_annotations
#print(response)
print('Labels:')
ques=""
'''for label in labels:
    #print(label.description)
    ques=ques+label.description
print(ques)'''
str1=""
str2=""
for text in labels:
        str1=str1+'#{}#'.format(text.description)
var=0;
for i in str1:
    if (i=='#' and var==1):
        break;
    if i=='#':
        var=1 
    str2+=i     
print(str2)
############################################
list1=str2.split("Q")
list2=str2.split("ANS")
for i in range(len(list1)-1):
    my_objects.append(QA(" ".join(list1[i+1][list1[i+1].find(":")+2:].splitlines()[0:len(list1[i+1][list1[i+1].find(":")+2:].splitlines())])," ".join(list2[i+1][list2[i+1].find(":")+2:].splitlines()[0:len(list2[i+1][list2[i+1].find(":")+2:].splitlines())]),i))
############################################
sample_rate = 48000
chunk_size = 2048
r = sr.Recognizer() 
mic_list = sr.Microphone.list_microphone_names() 
device_id=1
for i, microphone_name in enumerate(mic_list): 
    if microphone_name == mic_name: 
        device_id = i  
with sr.Microphone(device_index = device_id, sample_rate = sample_rate,  
                        chunk_size = chunk_size) as source: 
    r.adjust_for_ambient_noise(source)
    speak.Speak("Next Question")
#    speak.Speak(ques)
    speak.Speak(my_objects[5].question)
    speak.Speak(my_objects[5].actual_answer)
    winsound.Beep(1500, 750) 
    print ("Speak your answer")
    audio = r.listen(source) 
    winsound.Beep(1500, 750)    
    try: 
        text = r.recognize_google(audio)
        mytext =text
        #print ("you said: " + text) 
      
    except sr.UnknownValueError: 
        print("Google Speech Recognition could not understand audio") 
      
    except sr.RequestError as e: 
        print("Could not request results from Google Speech Recognition service; {0}".format(e)) 
#speak.Speak(mytext)
ans=mytext
act_ans=my_objects[5].actual_answer
'''count=0
words1=ans.split()
#print(words1)
words2=act_ans.split()
#print(words2)
for i in range(1,len(words1)):
    for j in range(1,len(words2)):
        if words1[i]==words2[j]:
            count=count+1
accuracy=100*(count+1)/len(words2)'''
speak.Speak("Your accuracy is "+str(fuzz.ratio(act_ans, ans))+" percent")
