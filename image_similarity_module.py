

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
import glob
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import ImageFont
from PIL import ImageDraw 
from PIL import Image


# Reading feature vector database i.e Recent updated database. it will select the most recently updated database
# Pass the path  
list_of_files = glob.glob('./Stream_1_Image_output_DB/*.csv') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
print(latest_file)
print("File will take time to load")
features_vector_df = pd.read_csv(latest_file)
features_vector_df = features_vector_df.drop('Unnamed: 0',axis=1)
print("Latest Feature Vector file has been read successfully")



# Inception model being initialised
model = InceptionV3(weights='imagenet', include_top=False)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Reading master file for insects information for example- family, order , genus and other info
new_data = pd.read_csv('./species_list/complete_species_list.csv')
print("Species list information has been read")

names = []
# Create a feature vector for test image which will be used for comapring with our database of images using cosine fnction.
# Model used is Inception V3 model
def creating_feature_vector(path):
    test_image_feature_list = []
    
    for images in os.listdir(path):
        names.append(images)
        try:
            img = image.load_img(os.path.join(path,images), target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)

            incep_feature = model.predict(img_data)
            incep_feature_np = np.array(incep_feature)
            test_image_feature_list.append(incep_feature_np.flatten())
        except:
            continue
    test_image_feature_list_nparray = np.array(test_image_feature_list) 
    print("Feature list created")
    return test_image_feature_list_nparray


temp_2=[]
for row in features_vector_df.iterrows():
    index, data = row
    #print(index, data)
    temp_2.append(data.tolist())

def cosine_check(path):
    test_image_feature_list_nparray = creating_feature_vector(path)
    print("Cosine comaprison is in progress")
    list_of_cosine_sim_factor = [] 

    for i in range(len(test_image_feature_list_nparray)):
        list1 = []
        for vector in temp_2:
            cosine_similarity_value = cosine_similarity([test_image_feature_list_nparray[i]], [vector])
            list1.append(cosine_similarity_value)
        
        list_of_cosine_sim_factor.append(list1)

    return list_of_cosine_sim_factor

# def get_images(folder_path,type_result, data):
#     subdir = 'T:\\projects\\2018\\SCION\\zzz_Ravi_Trash\\annot_images'
#     list_for_print = list(data['original_image_name'])
    
#     path = folder_path + '/' + type_result
#     os.mkdir(path)
#     data.to_excel(path + type_result + '.xlsx')
#     for images in list_for_print:
#         print(images)
#         img = cv2.imread(os.path.join(subdir,images))
#         filename = path + '/%s'%images
#         cv2.imwrite(filename,img) 
    


def concat_images(name,folder_path, type_result, data,path):
    subdir = './annot_images'
    path = path
    
    list_for_print = list(data['original_image_name'])
    imgs = [Image.open(os.path.join(path,name))]
    for images in list_for_print:
        img  = Image.open(os.path.join(subdir,images))
        imgs.append(img)
        
    min_shape = (256,256)

    list1= []
    for i in range(len(imgs)):
        if i == 0:
            resize_image = imgs[i].resize(min_shape)
            draw = ImageDraw.Draw(resize_image)
            #font = ImageFont.load("arial.pil")
            # font = ImageFont.truetype(<font-file>, <font-size>)
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 22, encoding="unic")
            # draw.text((x, y),"Sample Text",(r,g,b))
            name_new = str(name).split('_')
            draw.text((0, 0),"Target_image" + '\n' +name_new[0]+ ' '+name_new[1],font=font,
                     fill=(255,0,0,255))
            list1.append(np.asarray(resize_image))
            #font=font
        else: 
            resize_image = imgs[i].resize(min_shape)
            draw = ImageDraw.Draw(resize_image)
            #font = ImageFont.load("arial.pil")
            # font = ImageFont.truetype(<font-file>, <font-size>)
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 22, encoding="unic")
            # draw.text((x, y),"Sample Text",(r,g,b))
            text = str(data['Order'][i-1]) + '\n' + str(data['similarity_value'][i-1]) + '\n' + str(data['original_species_name'][i-1])
            draw.text((0, 0),text,font=font,fill=(255,0,0,255))
            list1.append(np.asarray(resize_image))
    imgs_comb = np.hstack(list1)
    imgs_comb = Image.fromarray( imgs_comb)
    filename = folder_path + '/{}_{}'.format(name,type_result)
    imgs_comb.save( filename +'.jpg')
   


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Image similarity using cosine')

    parser.add_argument('--path', required=False,
                        metavar="/path/to/insect/images/",
                        help='Directory of the Insect images')

    args = parser.parse_args()
    
    list_of_cosine_sim_factor = cosine_check(args.path)

    
    # We are extracting image name from image database so that we can later combine with the master file for similar images information.

    subdir = './annot_images'
    original_name = []
    image_name = []
    for images in os.listdir(subdir):
        original_name.append(images)
        item = images.split("_")
        if len(item) < 3:
            item = item[0]
            image_name.append(item)
            #print(item)
        else:
            new = images.split('_')
            #print(new[0] + ' ' + new[1])
            image_new = new[0] + ' ' + new[1]
            image_name.append(image_new)


    for i in range(len(list_of_cosine_sim_factor)):
        list1 = list_of_cosine_sim_factor[i]
        d= {'original_species_name':image_name,'similarity_value':list1, 'original_image_name':original_name}
        annot_insect_dataframe = pd.DataFrame(data = d)
        #print(annot_insect_dataframe.head())
    
        annot_insect_dataframe_new = annot_insect_dataframe.sort_values(by = 'similarity_value', ascending= False )
        #print(annot_insect_dataframe_new.head(15))
    
        annot_insect_df = annot_insect_dataframe_new.drop_duplicates(subset = 'original_species_name' ,keep = 'first')
        #print(annot_insect_df.head(15))
    
        insects_merged_data  = pd.merge(annot_insect_df, new_data, on= 'original_species_name', how = "inner")
        name = names[i]
        print(name)
        folder_path ='./test_result'
        #os.mkdir(folder_path)
    
#       img = cv2.imread(os.path.join(subdir,name))
#       filename = folder_path + '/%s'%name
#       cv2.imwrite(filename,img) 
    
        # now  to read the similar images
        result_images = insects_merged_data.head(10)
        result_images.to_excel("./test_result/image_info.xlsx")
        type_result = "normal_results"
        concat_images(name,folder_path, type_result, result_images,args.path)
        print("Image output can be senn in the following folder" + ' -- ' + folder_path)
        
        # now  to read the similar images based on order 
        top_3_order = insects_merged_data.head(10)
        d = top_3_order.groupby(by='Order', as_index=False).agg({'original_species_name': pd.Series.nunique})
        d = d.sort_values(by='original_species_name',ascending=False).reset_index()
        d = d.drop('index',axis=1)
        if d['original_species_name'][0] >1:
            result_with_top_order = insects_merged_data[insects_merged_data['Order']== d['Order'][0]]
        else:
            result_with_top_order = insects_merged_data[insects_merged_data['Order'] == top_3_order['Order'][0]]

        result_with_top_order = insects_merged_data[insects_merged_data['Order']== d['Order'][0]]
        result_with_top_order = result_with_top_order.reset_index()
        result_with_top_order.drop(['index','Unnamed: 0'], axis =1, inplace= True)
        result_with_top_order = result_with_top_order.head(10)
        #print(result_with_top_order)
        type_result = "same_order_results"

        concat_images(name, folder_path, type_result, result_with_top_order,args.path)
    


