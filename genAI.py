# import google.generativeai as genai
# import vertexai
# from vertexai.generative_models import GenerativeModel, ChatSession

# model = genai.GenerativeModel('gemini-1.5-flash')

# project_id = "learning-helper-2024"
# vertexai.init(project=project_id, location="us-central1")

# def get_chat_response(chat: ChatSession, prompt: str) -> str:
#     text_response = []
#     responses = chat.send_message(prompt, stream=True)
#     for chunk in responses:
#         text_response.append(chunk.text)
#     return "".join(text_response)

# chat = model.start_chat(history=[])

# print(get_chat_response(chat, 'Hi'))

import vertexai
from vertexai.generative_models import GenerativeModel

PROJECT_ID = "learning-helper-2024"
vertexai.init(project=PROJECT_ID, location="us-central1")

model = GenerativeModel("gemini-1.5-flash-002")

text="""hello
"""

prompt="make this text into json format, make Q into question attribute and ANS into answer attribute for all the rows:"
response1 = model.generate_content(
    prompt+text
)

print(response1.text)