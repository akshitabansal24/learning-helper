import vertexai
from vertexai.generative_models import GenerativeModel
import json

PROJECT_ID = "learning-helper-2025-451212"
vertexai.init(project=PROJECT_ID, location="us-central1")

model = GenerativeModel("gemini-1.5-flash-002")

def genQA(text):
    print("Received text:",text)
    AItext = text
    prompt1 = ( "Hello, you are a learning helper tutor. Your job is to help students in learning their study material."
                "You will be provided with some study material notes, in either question answer format"
                "or study matter notes, understand the text according to the content, "
                "and then construct relevant questions and answers related to the material in a proper JSON format,"
                "with question into 'Ques' attribute and answer into 'Ans' attribute for entire content:"
                "After you have created the JSON content, your response will automatically get written to a JSON file 'generated_QA.json'"
                "Ensure the response is a valid JSON object without enclosing it in triple backticks or markdown formatting:"
                )
    response1 = model.generate_content(
        prompt1+AItext
    )

    raw_text = response1.text.strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]  # Remove "```json\n"
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]  # Remove "```"

    try:
        json_data = json.loads(raw_text)
    #     with open("generated_QA.json", "w", encoding="utf-8") as file:
    #         json.dump(json_data, file, indent=4, ensure_ascii=False)
        print("QA pairs success")
        return json_data
    except json.JSONDecodeError:
        print("Error: Generated response is not a valid JSON format.")