

# Import necessary libraries
import sys
import numpy as np
import pandas as pd
import os
import keras
from keras.preprocessing import image 
from keras.applications.inception_v3 import InceptionV3 # Using Inception V3 model for our working
from keras.applications.inception_v3 import preprocess_input
from datetime import datetime
import time




#subdir = 'T:/projects/2018/SCION/zzz_Ravi_Trash/Test_Folder_01062019/similar_images_output'

model = InceptionV3(weights='imagenet', include_top=False)
model.compile(optimizer='adam', loss='categorical_crossentropy')


def creating_feature_vector(path):
    incep_feature_list = []
    for images in os.listdir(path):
        #print(images)
        img = image.load_img(os.path.join(path,images), target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        incep_feature = model.predict(img_data)
        incep_feature_np = np.array(incep_feature)
        incep_feature_list.append(incep_feature_np.flatten())

    
    incep_feature_list_nparray = np.array(incep_feature_list)
    return incep_feature_list_nparray



def creating_database(path):
    feature_list = creating_feature_vector(path)

    features_dict ={}
    for i  in range(len(feature_list)):
        features_dict[i] = feature_list[i]

    features_vector_df = pd.DataFrame(data=features_dict)
    features_vector_df = features_vector_df.T
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = "./Stream_1_Image_output_DB/"+timestr+"_"+"feature_vector_output.csv" 
    features_vector_df.to_csv(filename)
    print(" File has been uploaded successfully")

def updating_database(path2):
    # Reading feature vector database i.e Recent updated database. it will select the most recently updated database
    # Pass the path  
    list_of_files = glob.glob('./Stream_1_Image_output_DB/*.csv') # * means all and looking for specific format i.e. *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    existing_dataframe = pd.read_csv(latest_file)
    existing_dataframe = existing_dataframe.drop('Unnamed: 0',axis=1)
    feature_list = creating_feature_vector(path2)
    features_dict ={}
    for i  in range(len(feature_list)):
        features_dict[i] = feature_list[i]

    features_vector_df = pd.DataFrame(data=features_dict)
    features_vector_df = features_vector_df.T
    features_vector_df.to_csv("./Stream_1_Image_output_DB/intermediate_file.csv")
    intermediate_dataframe = pd.read_csv("./Stream_1_Image_output_DB/intermediate_file.csv")
    internediate_dataframe = intermediate_dataframe.drop('Unnamed: 0',axis=1)
    updated_dataframe = pd.concat([existing_dataframe,internediate_dataframe])
    updated_dataframe = updated_dataframe.reset_index()
    updated_dataframe.drop('index', axis=1, inplace=True)
    #filename = "M:/test_image_db/"+"{:%H:%M_%b_%d_%Y_}".format(datetime.now())+"feature_vector_output.csv"
    #filename = 'M:/test_image_db/feature_vector_output.csv'
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = "./Stream_1_Image_output_DB/"+timestr+"_"+"feature_vector_output.csv" 
    updated_dataframe.to_csv(filename)
    print(" File has been updated successfully")


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Creating Database')

    parser.add_argument('--path', required=False,metavar="/path/to/insect/dataset/",help='Directory of the Insect images')
    parser.add_argument('--path2', required= False, metavar="/path/to/directory/images", help='Directory of the Insect images for updating')

    args = parser.parse_args()
    if args.path:
        creating_database(args.path)
    elif args.path2:
        updating_database(args.path2)
    else:
        print("please give correct path for images")

