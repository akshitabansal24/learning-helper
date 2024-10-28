import vertexai
from vertexai.generative_models import GenerativeModel

PROJECT_ID = "ambient-union-440009-q4"
vertexai.init(project=PROJECT_ID, location="us-central1")

model = GenerativeModel("gemini-1.5-flash-002")

text="""hello
"""

prompt="make this text into json format, make Q into question attribute and ANS into answer attribute for all the rows:"
response1 = model.generate_content(
    prompt+text
)

print(response1.text)