#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import json
import pandas as pd
import itertools
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
import re
import ast
import matplotlib.pyplot as plt


# In[2]:


def create_dataframe(dataset_path):
    """Function used to create a Pandas DataFrame containing specifications page titles

    Reads products specifications from the file system ("dataset_path" variable in the main function) and creates a Pandas DataFrame where each row is a
    specification. The columns are 'source' (e.g. www.sourceA.com), 'spec_number' (e.g. 1) and the 'page title'. Note that this script will consider only
    the page title attribute for simplicity.

    Args:
        dataset_path (str): The path to the dataset

    Returns:
        df (pd.DataFrame): The Pandas DataFrame containing specifications and page titles
    """

    print('>>> Creating dataframe...\n')
    columns_df = ['source', 'spec_number', 'spec_id', 'page_title']

    progressive_id = 0
    progressive_id2row_df = {}
    for source in tqdm(os.listdir(dataset_path)):
        for specification in os.listdir(os.path.join(dataset_path, source)):
            specification_number = specification.replace('.json', '')
            specification_id = '{}//{}'.format(source, specification_number)
            with open(os.path.join(dataset_path, source, specification)) as specification_file:
                specification_data = json.load(specification_file)
                page_title = specification_data.get('<page title>').lower()
                row = (source, specification_number, specification_id, page_title)
                progressive_id2row_df.update({progressive_id: row})
                progressive_id += 1
    df = pd.DataFrame.from_dict(progressive_id2row_df, orient='index', columns=columns_df)
    print('>>> Dataframe created successfully!\n')
    return df


# In[3]:


df = create_dataframe('../datasets/unlabeled/2013_camera_specs')


# ## Title

# In[4]:


df.head()


# In[5]:


df = df.drop(columns = ["source", "spec_number"], axis = 1)


# In[6]:


from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))


# In[7]:


stopWords = set(['itself', 'down', 'by', 'with', 'doesn', 'wouldn', 'other', 'ours', 'of', 'then', 'where', 'don', 'these', 'nor', 'she', "should've", 'won', 'ma', 'from', 'had', "you're", 'our', 'did', 'them', 'too', 'her', 'that', 'haven', 'after', "you'll", 'hers', 'because', 'yourself', 'against', 'mightn', 'as', 'll', 'whom', 'how', 'couldn', 'further', 'aren', "you'd", 'and', 'needn', "couldn't", 'those', 'to', "doesn't", "weren't", 'both', 'ourselves', 'in', 'which', 'yours', 'under', 'some', 'what', 'during', 'before', "needn't", "shan't", 'here', 'having', 'hasn', 'your', "hasn't", 'between', 'me', "she's", 'into', 'all', 'at', 'shan', 'who', 'o', 'an', 'very', 'can', 'you', 'shouldn', 'such', 'but', 'do', 'out', 'am', "shouldn't", 'above', 'wasn', 'or', 'were', 'own', 'didn', "you've", 'on', 'will', 'my', 'it', 'have', 'once', 'only', 'been', 'themselves', 'his', 'be', "mightn't", 'they', 'not', 'so', 'up', 'any', 'most', 'has', 'myself', 't', 'yourselves', 'isn', "it's", 'y', 'm', 'now', 'until', 're', 'there', 'their', 'mustn', "mustn't", 'again', 'being', 'hadn', 'doing', 'just', 'no', 'if', 've', "wasn't", "won't", 'we', 'below', 'does', 'more', 'this', 'should', "isn't", 'ain', "don't", 'i', "haven't", 'than', "didn't", 'are', 'about', 'off', 'him', 'for', 'few', "wouldn't", 'was', 'weren', 'why', 'he', "that'll", 'd', 'the', 'its', 'a', 'each', 'is', 'while', "aren't", 'when', 'theirs', 'same', 's', 'himself', 'herself', "hadn't", 'through', 'over'])


# In[8]:


punctuation = "!#$%&'()*+,-./:;<=>?@[\]^_`{|}~€£¥₹₽"


# In[9]:


def replace_punctuation(word):
    return ''.join(c for c in word if c not in punctuation)


# In[10]:


df["page_title"] = df["page_title"].apply(lambda x : [i.lower() for i in list(map(lambda y: replace_punctuation(y), word_tokenize(x))) if i and i.lower() not in stopWords])


# ### Modelwords

# In[11]:


pattern = re.compile("(\S*[A-Za-z]\S*[0-9]\S*|\S*[0-9]\S*[A-Za-z]\S*)")


# In[12]:


## In the data replace lumix with panasonic


# In[13]:


brands = ['360fly', 'acer', 'achiever', 'acorn', 'actionpro', 'activeon', 'aee', 'agfa', 'agfaphoto', 'aiptek', 'akaso', 'alpine', 'alpine', 'amkov', 'andoer', 'annke', 'ansco', 'apeman', 'apex', 'apple', 'archos', 'argus', 'arlo', 'arri', 'axis', 'bell', 'benq', 'blackmagic', 'blackmagic', 'bosch', 'bower', 'brinno', 'brookstone', 'browning', 'cambo', 'campark', 'canon', 'carl', 'casio', 'celestron', 'chinon', 'cisco', 'cobra', 'coleman', 'concord', 'contax', 'contour', 'covert', 'craig', 'crayola', 'creative', 'creative', 'crosstour', 'crumpler', 'datavideo', 'delkin', 'dell', 'digitrex', 'discovery', 'disney', 'dji', 'd-link', 'domke', 'dörr', 'dragon', 'dxg', 'dxo', 'easypix', 'elecom', 'elmo', 'emerson', 'energizer', 'epson', 'fisher-price', 'flip', 'flir', 'foscam', 'fotoman', 'fotopro', 'fuji', 'fujifilm', 'fujinon', 'garmin', 'gateway', 'godox', 'goodmans', 'google', 'gopro', 'grundig', 'hahnel', 'hamilton', 'hasselblad', 'hello', 'herofiber', 'hitachi', 'holga', 'horseman', 'hoya', 'htc', 'huawei', 'ikelite', 'ilford', 'impossible', 'innovage', 'insignia', 'insta360', 'intel', 'intova', 'ion', 'iris', 'jazz', 'jenoptik', 'jjrc', 'jvc', 'kaiser', 'kenko', 'keyence', 'king', 'kitvision', 'kodak', 'konica', 'kyocera', 'leaf', 'lego', 'leica', 'lenovo', 'lexibook', 'linhof', 'liquid', 'little', 'logitech', 'lomography', 'lowepro', 'ltl', 'lytro', 'maginon', 'magnavox', 'mamiya', 'manfrotto', 'marshall', 'marumi', 'mattel', 'maxell', 'meade', 'medion', 'memorex', 'mercury', 'metz', 'microsoft', 'microtek', 'midland', 'minolta', 'minox', 'monster', 'motorola', 'moultrie', 'mustek', 'nabi', 'neewer', 'nest', 'netgear', 'night', 'nikkon', 'nikkor', 'nikon', 'nilox', 'nintendo', 'nippon', 'nokia', 'norcent', 'olympus', 'optech', 'ordro', 'oregon', 'packard', 'palm', 'panasonic', 'parrot', 'pelco', 'pentacon', 'pentax', 'phase', 'philips', 'philips', 'phoenix', 'pioneer', 'playskool', 'polaroid', 'polarpro', 'praktica', 'premier', 'promaster', 'proscan', 'pyle', 'radioshack', 'raymarine', 'raynox', 'rca', 'ricoh', 'ring', 'rode', 'rokinon', 'rollei', 'ryobi', 'sakar', 'samsung', 'sandisk', 'sanyo', 'schneider', 'schneider', 'schneider', 'scosche', 'seasea', 'sealife', 'sharp', 'sharper', 'sigma', 'sinar', 'sipix', 'sjcam', 'sony', 'soocoo', 'stealth', 'superheadz', 'svp', 'swann', 'tamrac', 'tamron', 'technika', 'tenba', 'think', 'thule', 'tokina', 'tomy', 'toshiba', 'transcend', 'traveler', 'trust', 'verbatim', 'vibe', 'victure', 'vistaquest', 'vivitar', 'voigtländer', 'vtech', 'vupoint', 'walimex', 'wyze', 'xiaomi', 'xit', 'xtreme', 'yashica', 'zeiss']


# In[14]:


df["page_title"] = df["page_title"].apply(lambda line : list(set(filter(lambda word : bool(pattern.match(word)) or word in brands,line))))


# In[15]:


df.head()


# In[16]:


df["brand"] = [[] for _ in range(len(df))]


# In[17]:


# See how many products have more than one brand
for index, row in df.iterrows():
    for brand in row["page_title"]:
        if brand in brands:
            df.at[index, "brand"].append(brand)
            row["page_title"].remove(brand)


# In[18]:


def clean_mp_mm_g_oz(value):
    if not isinstance(value, list) and pd.isna(value):
        return np.nan
    regex = r"[0-9]+mm(\n|)"
    regex2 = r"[0-9]+mp(\n|)"
    regex3 = r"[0-9]+oz"
    regex4 = r"[0-9]+g(\n|)$"
    repl = value
    for e in repl:
        if bool(re.match(regex, e)) or bool(re.match(regex2, e)) or bool(re.match(regex3, e)) or bool(re.match(regex4, e)):
            repl.remove(e)
    return repl


# In[19]:


df["page_title"] = df["page_title"].apply(lambda row : clean_mp_mm_g_oz(row))


# In[20]:


df.head()


# ## Load cleaned datasets

# In[21]:


import os
import glob

os.chdir("../datasets/unlabeled/cleaned")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
df_cleaned = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv


# In[22]:


df_cleaned = df_cleaned.reset_index(drop = True)


# In[23]:


df_cleaned.drop(columns=["page_title"], inplace=True)


# In[24]:


df_cleaned.head()


# ## Merge clean with title

# In[25]:


df = df.merge(df_cleaned, on="spec_id")


# In[26]:


df.head()


# In[27]:


df.rename(columns={"brand_x" : "brand_from_title", "brand_y" : "brand_descr"}, inplace=True)


# In[28]:


df.head()


# In[29]:


def clean_short_descr(line):
    pattern = re.compile("(\S*[A-Za-z]\S*[0-9]\S*|\S*[0-9]\S*[A-Za-z]\S*)")
    brands = ['360fly', 'acer', 'achiever', 'acorn', 'actionpro', 'activeon', 'aee', 'agfa', 'agfaphoto', 'aiptek', 'akaso', 'alpine', 'alpine', 'amkov', 'andoer', 'annke', 'ansco', 'apeman', 'apex', 'apple', 'archos', 'argus', 'arlo', 'arri', 'axis', 'bell', 'benq', 'blackmagic', 'blackmagic', 'bosch', 'bower', 'brinno', 'brookstone', 'browning', 'cambo', 'campark', 'canon', 'carl', 'casio', 'celestron', 'chinon', 'cisco', 'cobra', 'coleman', 'concord', 'contax', 'contour', 'covert', 'craig', 'crayola', 'creative', 'creative', 'crosstour', 'crumpler', 'datavideo', 'delkin', 'dell', 'digitrex', 'discovery', 'disney', 'dji', 'd-link', 'domke', 'dörr', 'dragon', 'dxg', 'dxo', 'easypix', 'elecom', 'elmo', 'emerson', 'energizer', 'epson', 'fisher-price', 'flip', 'flir', 'foscam', 'fotoman', 'fotopro', 'fuji', 'fujifilm', 'fujinon', 'garmin', 'gateway', 'godox', 'goodmans', 'google', 'gopro', 'grundig', 'hahnel', 'hamilton', 'hasselblad', 'hello', 'herofiber', 'hitachi', 'holga', 'horseman', 'hoya', 'htc', 'huawei', 'ikelite', 'ilford', 'impossible', 'innovage', 'insignia', 'insta360', 'intel', 'intova', 'ion', 'iris', 'jazz', 'jenoptik', 'jjrc', 'jvc', 'kaiser', 'kenko', 'keyence', 'king', 'kitvision', 'kodak', 'konica', 'kyocera', 'leaf', 'lego', 'leica', 'lenovo', 'lexibook', 'linhof', 'liquid', 'little', 'logitech', 'lomography', 'lowepro', 'ltl', 'lytro', 'maginon', 'magnavox', 'mamiya', 'manfrotto', 'marshall', 'marumi', 'mattel', 'maxell', 'meade', 'medion', 'memorex', 'mercury', 'metz', 'microsoft', 'microtek', 'midland', 'minolta', 'minox', 'monster', 'motorola', 'moultrie', 'mustek', 'nabi', 'neewer', 'nest', 'netgear', 'night', 'nikkon', 'nikkor', 'nikon', 'nilox', 'nintendo', 'nippon', 'nokia', 'norcent', 'olympus', 'optech', 'ordro', 'oregon', 'packard', 'palm', 'panasonic', 'parrot', 'pelco', 'pentacon', 'pentax', 'phase', 'philips', 'philips', 'phoenix', 'pioneer', 'playskool', 'polaroid', 'polarpro', 'praktica', 'premier', 'promaster', 'proscan', 'pyle', 'radioshack', 'raymarine', 'raynox', 'rca', 'ricoh', 'ring', 'rode', 'rokinon', 'rollei', 'ryobi', 'sakar', 'samsung', 'sandisk', 'sanyo', 'schneider', 'schneider', 'schneider', 'scosche', 'seasea', 'sealife', 'sharp', 'sharper', 'sigma', 'sinar', 'sipix', 'sjcam', 'sony', 'soocoo', 'stealth', 'superheadz', 'svp', 'swann', 'tamrac', 'tamron', 'technika', 'tenba', 'think', 'thule', 'tokina', 'tomy', 'toshiba', 'transcend', 'traveler', 'trust', 'verbatim', 'vibe', 'victure', 'vistaquest', 'vivitar', 'voigtländer', 'vtech', 'vupoint', 'walimex', 'wyze', 'xiaomi', 'xit', 'xtreme', 'yashica', 'zeiss']
    if not isinstance(line, list) and pd.isna(line):
        return np.nan
    else:
        line = ast.literal_eval(line)
        return list(set(filter(lambda word : bool(pattern.match(word)) or word in brands,line)))


# In[30]:


df["short_descr"] = df["short_descr"].apply(clean_short_descr)


# In[31]:


df["short_descr"] = df["short_descr"].apply(lambda row : clean_mp_mm_g_oz(row))


# In[32]:


df.head()


# ## Add units to megapixels and screen_size

# In[33]:


df["megapixels"] = df["megapixels"].apply(lambda value: str(value) + "mp" if not pd.isna(value) else np.nan)


# In[34]:


df["screen_size"] = df["screen_size"].apply(lambda value: str(value) + "in" if not pd.isna(value) else np.nan)


# In[35]:


df["weight"] = df["weight"].apply(lambda value: str(value) + "g" if not pd.isna(value) else np.nan)


# In[36]:


df.head()


# In[37]:


df.isna().sum() / len(df)


# In[38]:


len(df)


# In[39]:


def create_brands_column(row):
    repl = row["brand_from_title"]
    if not pd.isna(row["brand_descr"]):
        repl.append(row["brand_descr"])
    if not pd.isna(row["manufacturer"]):
        repl.append(row["manufacturer"])
    return tuple(set(repl))


# In[40]:


df["merged_brands"] = df.apply(create_brands_column, axis = 1)


# In[41]:


df.head()


# In[42]:


sum(df.apply(lambda row : row["page_title"] == [], axis = 1))


# In[43]:


df.drop(columns = ["brand_from_title", "brand_descr", "manufacturer"], inplace=True)


# In[44]:


grouped = df.groupby("merged_brands")


# In[45]:


unbranded_til_100 = grouped.get_group(())


# In[46]:


for gname, group in grouped:
    if len(group) < 100:
        unbranded_til_100 = pd.concat([group, unbranded_til_100])


# In[47]:


def get_merged_df(dataframe):
    merged = dataframe.drop(columns=["merged_brands"], axis = 1)
    merged = (merged.merge(merged, on=merged.assign(key_col=1)['key_col'], suffixes=('', '_right'))
 .query('spec_id < spec_id_right') # filter out joins on the same row and keep unique combinations
 .reset_index(drop=True))
    merged.drop(columns = ["key_0"], axis = 1, inplace=True)
    merged.rename(columns = {"spec_id" : "left_spec_id", "spec_id_right" : "right_spec_id"}, inplace=True)
    merged.reset_index(inplace=True)
    return merged


# In[49]:


def determine_match(row):
    if row["page_title"] == []:
        target = 0.55
    else:
        target = 1.3
        
    #print(row.isna().sum(), len(row))
    
    #not_nan_per = row.isna().sum() / len(row)

    
    dim_weight = 0.95
    dots_weight = 0.98
    mp_weight = 0.44
    scr_weight = 0.54
    type_weight = 0.49
    weight_weight = 0.81
    descr_weight= 0.94
    title_weight = 1.2
    
    
    score = 0
    dim_l = row["dimensions"]
    dim_r = row["dimensions_right"]
    dots_l = row["dots"]
    dots_r = row["dots_right"]
    megapixels_l = row["megapixels"]
    megapixels_r = row["megapixels_right"]
    screen_size_l = row["screen_size"]
    screen_size_r = row["screen_size_right"]
    short_descr_l = row["short_descr"]
    short_descr_r = row["short_descr_right"]
    type_l = row["type"]
    type_r = row["type_right"]
    weight_l = row["weight"]
    weight_r = row["weight_right"]
    page_title_l = row["page_title"]
    page_title_r = row["page_title_right"]
    
    dimensions_regex = r"([0-9]+\.[0-9]+|[0-9]+)h([0-9]+\.[0-9]+|[0-9]+)w([0-9]+\.[0-9]+|[0-9]+)d"
    dimensions_regex_2 = r"h([0-9]+\.[0-9]+|[0-9]+)w([0-9]+\.[0-9]+|[0-9]+)d([0-9]+\.[0-9]+|[0-9]+)"
    
    if not pd.isna(dim_l) and not pd.isna(dim_r):
        if re.match(dimensions_regex, dim_l) == None:
            groups_l = re.match(dimensions_regex_2, dim_l).groups(1)
        else:
            groups_l = re.match(dimensions_regex, dim_l).groups(1)
        if re.match(dimensions_regex, dim_r) == None:
            groups_r = re.match(dimensions_regex_2, dim_r).groups(1)
        else:
            groups_r = re.match(dimensions_regex, dim_r).groups(1)
        if np.sum(np.abs(np.array(groups_l).astype(float) - np.array(groups_r).astype(float))) <= 0.3:
            score += 0.95
    if not pd.isna(dots_l) and not pd.isna(dots_r) and dots_l == dots_r:
        score += 0.98
    if not pd.isna(megapixels_l) and not pd.isna(megapixels_r) and abs(float(megapixels_l.replace("mp", "")) - float(megapixels_r.replace("mp", ""))) <= 0.2:
        score += 0.44
    if not pd.isna(screen_size_l) and not pd.isna(screen_size_r) and abs(float(screen_size_l.replace("in", "")) - float(screen_size_r.replace("in", ""))) <= 0.2:
        score += 0.54
    if not pd.isna(type_l) and not pd.isna(type_r) and type_l == type_r:
        score += 0.5
    if not pd.isna(weight_l) and not pd.isna(weight_r) and abs(float(weight_l.replace("g", "")) - float(weight_r.replace("g", ""))) <= 0.2:
        score += 0.81
        
    if isinstance(page_title_r, list) and isinstance(short_descr_l, list) and short_descr_l == short_descr_r:
        for spec1 in short_descr_l:
            for spec2 in short_descr_r:
                if spec1 == spec2:  
                    score += 0.94
    if isinstance(page_title_r, list) and isinstance(page_title_l, list) and page_title_r == page_title_l:
        for spec1 in page_title_l:
            for spec2 in page_title_r:
                if spec1 == spec2:  
                    score += 1.2
   
    return score >= target
    


# In[50]:


for gname, group in grouped:
    labels = []
    if len(group) == 1 or gname == ():
        continue
 
    #brand_and_unbranded = pd.concat([group, unbranded])
    
    print("CALCULATING FOR BRAND = ", gname)
    merged = get_merged_df(group)

    #logic
    
    print("NUMBER OF COMPARISONS: ", len(merged))
    labels.append(list(merged.apply(determine_match, axis = 1)))
    labels = sum(labels, [])
    merged["label"] = labels
    print("MATCHED ", sum(merged["label"]), " OUT OF ", len(merged["label"]))
    del labels
    merged = merged.loc[merged['label'] == True]
    cols = ["left_spec_id", "right_spec_id"]
    merged = merged[cols]
    merged.to_csv("/Users/gfotiadis/programming/sigmod/datasets/created/with_details/{}_matches_labeled.csv".format(gname[0].replace("/", "")), index = False)
     


# In[51]:


labels_unbranded = []
merged_unbranded = get_merged_df(unbranded_til_100)
labels_unbranded.append(list(merged_unbranded.apply(determine_match, axis = 1)))
labels_unbranded = sum(labels_unbranded, [])
merged_unbranded["label"] = labels_unbranded
del labels_unbranded
merged_unbranded = merged_unbranded.loc[merged_unbranded['label'] == True]
cols = ["left_spec_id", "right_spec_id"]
merged_unbranded = merged_unbranded[cols]
merged_unbranded.to_csv("/Users/gfotiadis/programming/sigmod/datasets/created/with_details/unbranded_matches_labeled.csv", index = False)


# In[52]:


import os
import glob

os.chdir("/Users/gfotiadis/programming/sigmod/datasets/created/with_details/")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
combined_csv = pd.concat([pd.read_csv(f, header = 0) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "/Users/gfotiadis/programming/sigmod/datasets/created/with_details/combined_csv.csv", index=False, encoding='utf-8-sig')


# In[53]:


len(combined_csv)


# In[ ]:


# old 102476


# In[ ]:


# target 381,212


# In[ ]:




