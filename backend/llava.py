from typing import Union
from pydantic import BaseModel
from PIL import Image  
import PIL
from fastapi import FastAPI, File, UploadFile
import base64
import json
import requests
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()
 
llm = ChatOllama(model="llama3")


@app.get("/")
def read_root():
    return {
        "Hello": "World"
        }


@app.post("/lava/")
async def create_upload_file(file: UploadFile):
    
    img = Image.open(file.file)
    img.save("test.jpg")

    with open(r"test.jpg", "rb") as image:
        image_string = base64.b64encode(image.read()).decode("utf-8")
        
    url = 'http://localhost:11434/api/generate'
    data = {
        "model": "llava",
        "prompt": "Refer the image with one of these emotions Happy, Angry, Excited, Stressed.",
        'images' : [image_string]
    }

    response = requests.post(url, json=data, stream = True)
    text = ""

    if response.status_code == 200:
        try:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    t = chunk.decode('utf-8')
                    t = json.loads(t)
                    text += t["response"]
        except Exception as e:
            print("Error reading response:", e)
    
    else:
        return {"Error": response.text}
    template_expression = """
                You are a tutor who understands the emotion of the student.
                  If he is happy, encourage him more. 
                  If the student is angry, then tell the student not to get angry for scoring wrong answers.
                  If he is sad, encourage him to write the next quiz well. 
                  If he is stressed, motivate him and try to Conversate according to his mood.
                  Provide the response in short.

                emotion : {emotion}
    """

    prompt = ChatPromptTemplate.from_template(template_expression)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"emotion":text})
    # return {"sample":"sample"}



 


 