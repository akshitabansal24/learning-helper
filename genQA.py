import vertexai
from vertexai.generative_models import GenerativeModel
import json
import os

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
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

def checkAnswer(userAnswer, correctAnswer):
    print("User Answer:", userAnswer)
    print("Correct Answer:", correctAnswer)

    user_answer = userAnswer
    correct_answer = correctAnswer

    prompt2 = ("Hello, you are an AI-powered learning tutor designed to help students improve their study material understanding." 
                
                "You will be provided with two answers:"
                "1. The correct answer, which is taken from the study material."  
                "2. The student answer, which is the response given by the student based on their understanding." 

                "- Compare the student answer with the correct answer ignoring capitalization, punctuation, and minor spelling errors as text is picked up from speech recognition so there may be those errors.."  
                "- Evaluate the student response based on the following assessment criteria:"  
                "1. Meaning & Accuracy (%): How well the student answer conveys the intended meaning of the correct answer based on main keywords. " 
                "2. Fluency & Coherence (%): How smoothly the student expresses the answer in a well-structured manner.  "
                "3. Presentation & Clarity (%): Whether the response is clear, logical, and easy to follow.  "
                "4. Key Information Coverage (%): How many essential points from the correct answer are covered in the student response. " 
                "5. Grammar & Expression (%): How well the student formulates the answer, excluding minor spelling errors.  "

                "Provide the response as a valid JSON object. The JSON should have the following structure: "
                "{'Evaluation': { 'Meaning_Accuracy': 'X%', 'Fluency_Coherence': 'X%', 'Presentation_Clarity': 'X%','Key_Information_Coverage': 'X%', 'Grammar_Expression': 'X% },"
                "'Feedback': {'Strengths': 'List of what the student did well.','Areas_for_Improvement': 'List of aspects where the student can improve.','Suggested_Changes': 'Provide actionable feedback on how the student can enhance their answer.'},"
                "'Overall_Score': 'Final percentage score based on overall performance.'}"

                "Hello, you are a learning helper tutor. Your job is to help students in learning their study material."
                "You will be provided with two answers, first is actual correct answer, taken from study material"
                "and second is user answer given in as input by student as per his learning and understanding."
                "You job is to compare those the student answer with correct answer in terms of meaning, accuracy, fluency, presentation,"
                "and any other metrics also if you find fit. And tell me these statistics in percentage criteria as marks."
                "Give detailed marksheet and suggest feedback and changes on how the student can enhance his output."
                "Ensure the response is a valid JSON object without enclosing it in triple backticks or markdown formatting:"
                )
                
    response2 = model.generate_content(
        prompt2+correct_answer+user_answer
    )

    raw_text = response2.text.strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]  # Remove "```json\n"
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]  # Remove "```"

    try:
        json_data = json.loads(raw_text)
    #     with open("generated_QA.json", "w", encoding="utf-8") as file:
    #         json.dump(json_data, file, indent=4, ensure_ascii=False)
        print("Feedback success")
        return json_data
    except json.JSONDecodeError:
        print("Error: Generated response is not a valid JSON format.")