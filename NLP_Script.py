#Python 3.7.2
# Used to analyze individual concatenated texts with Watson NLP using IBM Cloud
import csv
import os
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions
import pandas as pd

df = pd.read_csv("./SentenceCompletionTests_raw.csv", encoding='latin1')
stems = df.iloc[0] # get values of the first row (the stems)
df = df.iloc[1:] # drop the first row from data frame (exclude the stems)

# for loop below concatenates the stem with each answer response
for i,col in enumerate(df.columns):
    #firstCol = df.iloc[:, 0]
    #df = df.drop(['ResponseId'], axis=1)
    # skip the first column (ResponseId)
    if i > 0:
        # update value to include the appropriate stem
        # the stems index matches with the column index
        df.loc[:,col] = stems[i] + " " + df.loc[:,col]

# TODO: Handle cases with multiple keywords and no keywords
def write_to_csv(name, results):
    file_path = name + ".csv"
    
    # Delete result files, if there are any
    if os.path.exists(file_path):
        os.remove(file_path)
        
    file = open(file_path, 'w', newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerow(["Sentiment", "Sadness", "Joy", "Fear", "Disgust", "Anger"])
    
    for data in results:   
        if len(data["keywords"]) > 0:
            values = [data["keywords"][0]["sentiment"]["score"]]
            values.append(data["keywords"][0]["emotion"]["sadness"])
            values.append(data["keywords"][0]["emotion"]["joy"])
            values.append(data["keywords"][0]["emotion"]["fear"])
            values.append(data["keywords"][0]["emotion"]["disgust"])
            values.append(data["keywords"][0]["emotion"]["anger"])
            # print(values)
        else:
            values = ["No data"] * 6
        csv_writer.writerow(values)
    file.close()
    print("File " + file_path + " is complete");
    
#authenticate nlp set service for Watson
authenticator = IAMAuthenticator('izOZcF-yOaL5TEyRQQX5en3HtNYKlEUaoYTyhwJ3gnVv')
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2020-08-01',
    authenticator=authenticator)

natural_language_understanding.set_service_url('https://api.us-south.natural-language-understanding.watson.cloud.ibm.com')

# Iterate over each participant
for index, row in df.iterrows():
    i = 0
    results = []
    # Iterate over each answer for a given participant
    for col in row:
        if i == 0: # Get the participant's name
            name = col
            i += 1
            continue
        
        response = natural_language_understanding.analyze(language='en',
                    text = col,
                    features=Features(entities=EntitiesOptions(emotion=True, sentiment=True, limit=2), keywords=KeywordsOptions(emotion=True, sentiment=True, limit=2))).get_result()
        results.append(response)
    write_to_csv(name, results)

