import json
import os

def main():
    os.chdir("scripts/LabExamples/")
    print(os.getcwd())
    with open("image_text.json", 'r') as f:
                json_data = json.load(f)

    text_arr = []
    for i in json_data:
        for text in i['text']:  
            text_arr.append(text[1])
            print(text[1])
    

main()