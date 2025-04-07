# from vertexai.preview.generative_models import GenerativeModel,  Image
from google import generativeai as  genai
from PIL import Image

GOOGLE_API_KEY="AIzaSyAxnVOuLEjDev8Zy-Oz_H5l-yXVDKq7Dm0"
genai.configure(api_key=GOOGLE_API_KEY)

def generate_text(img) -> str:
    #load the model
    model = genai.GenerativeModel('gemini-1.5-flash')

    image = Image.open(img)

    #provide image and prompt to extract text
    response = model.generate_content(
        [ image,
         f'''
            You are an expert in number plate recognition:
            return the number plate(s)
         '''
        ]
    )

    return response.text

print(generate_text('./detected/nplates.jpeg')) 