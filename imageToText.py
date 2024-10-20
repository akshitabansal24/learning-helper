
# Imports the Google Cloud client library
from google.cloud import vision


def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    str1=""
    str2=""
    for text in texts:
            str1=str1+'#{}#'.format(text.description)
    var=0;
    for i in str1:
        if (i=='#' and var==1):
            break;
        if i=='#':
            var=1 
        str2+=i     
    print(str2)
    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

detect_text("test.jpeg")