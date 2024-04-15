import pandas as pd
import json 

def create_df(obj_json,sktl_json,txt_json):
    obj_json_df = pd.read_json(obj_json)
    obj_json_df.set_index('image')
    with open(sktl_json, 'r') as f:
        sktl_json_data = json.load(f)

    sktlDf = pd.DataFrame([{}])


    for x in sktl_json_data:
        personX = x['image']
        if 'body_parts' in x['skeltal_data'].keys():
            xSktlData = x['skeltal_data']['body_parts']
        else:
            xSktlData = {}
        temp_df = pd.DataFrame([xSktlData]).T
        temp_df.columns = [personX]
        

        sktlDf = pd.concat([sktlDf, temp_df], axis=1)

    sktlDf = sktlDf.iloc[1:].T

    sktlDf.index.name = 'image'

    text_json_df = pd.read_json(txt_json)

    text_json_df.set_index("image")

    
    final_df = pd.merge(text_json_df, obj_json_df, on='image', how='outer')


    final_df = pd.merge(final_df, sktlDf, on='image', how='outer')

  

    final_df = final_df.sort_values(by='idNo', ascending=True)
    final_df = final_df.reset_index()



    return final_df