import importlib
import os
import openai
from datetime import datetime
from dotenv import load_dotenv

openai.api_key = "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"



def call():

    user_msg = "I need a long-endurance unmanned aerial vehicle (UAV) optimized for reconnaissance and surveillance missions over remote areas. The aircraft should be capable of autonomous flight for extended periods, with a focus on stability, endurance, and payload capacity. It should have a range of at least 5,000 kilometers and be equipped with advanced sensors for intelligence gathering."


    system_msg2 = """you will receive an input that will consist of mission specification and requirements for a aircraft. your job is to take the information in that input and output a description that will be fed into an AI agent that will generate the cad model for the aircraft's wing. 
    based on the input, there is only 4 parameters you can vary:
    - NACA airfoil (only 4 digit)
    - aspect ratio
    - taper ratio
    - sweep angle

    here is an example of how your response should look like:

    "generate a CAD model of a wing using the NACA(insert 4 digit NACA serie) airfoil, with an aspect ratio of X, a taper ratio of X, and a sweep angle of X"



    Dont Include anything else in your response, since your output will be directly fed into the cad generation system. 
    """

    response2 = openai.ChatCompletion.create(
        model="gpt-4-turbo-2024-04-09",
        messages=[
            {"role": "system", "content": system_msg2},
            {"role": "user", "content": user_msg},
        ],
    )

    print(response2["choices"][0]["message"]["content"])





call()