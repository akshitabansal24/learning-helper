import cleanImage
import imageToText
import handwritingToText
import genQA 
import textToSpeech
import speechToText

printImage = 'test.jpeg'
handwritingImage = 'notes.png'

choice = input("Handwritten(1) or typed(2): ")
text = ''

if choice == "1":  
    image = handwritingImage
    # cleanedImage = cleanImage.cleanImage(image)
    text = handwritingToText.detectDocument(image)
else:
    image = printImage
    cleanedImage = cleanImage.cleanImage(image)
    text = imageToText.detect_text(cleanedImage)

# Generate QA based on detected text
genQA.genQA(text)  

textToSpeech.createAudio()
speechToText.listenAudio()
# evaluate each user ans with QA ans
