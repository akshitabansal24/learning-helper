import vertexai
from vertexai.generative_models import GenerativeModel

PROJECT_ID = "learning-helper-2024"
vertexai.init(project=PROJECT_ID, location="us-central1")

model = GenerativeModel("gemini-1.5-flash-002")

text="""
hello
"""

prompt="i have provided you some text, understand it, and identify most important set of 10 questions from it, that can come in exam point of view, also the answers for those questions, and then make that set of ques and answer text into json format, into question attribute and answer attribute for all the questions:"
response1 = model.generate_content(
    prompt+text
)

print(response1.text)