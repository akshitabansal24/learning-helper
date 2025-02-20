
# Imports the Google Cloud client library
from google.cloud import vision


def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    image_context = vision.ImageContext(language_hints=["en", "hi", "fr", "es", "zh"]) 

    response = client.text_detection(image=image, image_context=image_context)
    texts = response.text_annotations
    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    extracted_text = ""
    if texts:
        extracted_text = texts[0].description  # Extract full text

    return extracted_text
    # print(extracted_text)

# detect_text("hindi.png")