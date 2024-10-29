import cleanImage
import handwritingToText
import genAI
import textToSpeech
import speechToText

cleanImage.cleanImage()
handwritingToText.detectDocument('test.jpeg')
genAI.genQA()
# randomly ask ques from QA
textToSpeech.createAudio()
speechToText.listenAudio()
# evaluate each user ans with QA ans
