
# coding: utf-8

# In[46]:

#Merge puppy and status code , text only

import pandas as pd
import numpy as np
import string
import csv
import sys
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[47]:

puppy_data= pd.read_csv('TextandNumeric.csv')

trainer_data= pd.read_csv('TrainerInfo.csv')

puppytrainer_data = pd.read_csv('PuppyTrainerOutcome.csv')



# In[48]:

puppy_data.head(5)


# In[49]:

puppy_data.shape
df_train = pd.DataFrame(trainer_data, columns = ['dog_DogID', 'DayInLife'])
df_train.head(4)

df_puppytrainer = pd.DataFrame(puppytrainer_data, columns = ['dog_DogID','dog_SubStatusCode'])
df_puppytrainer.head(4)


# In[50]:

merge_puppy_and_puptrainer = pd.merge(puppy_data, df_puppytrainer,  how='inner', left_on='ogr_DogID', right_on='dog_DogID')
del merge_puppy_and_puptrainer['dog_DogID']


# In[51]:

merge_puppy_and_puptrainer.shape

merge_puppy_and_puptrainer['dog_SubStatusCode']= merge_puppy_and_puptrainer.loc[:,'dog_SubStatusCode'].replace([1,2,3,17,21,27,28,29,37,45,50,66,76,97,111,154,159,160,161,162,166,167,171,172,173,174,175,179,181,184,187,188],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
merge_puppy_and_puptrainer['dog_SubStatusCode']=merge_puppy_and_puptrainer.loc[:,'dog_SubStatusCode'].replace([23,25,26,27,55,98,99,121,169],[1,1,1,1,1,1,1,1,1])

#merge_puppy_and_puptrainer.head(3)


# In[52]:

final_merge = pd.merge(merge_puppy_and_puptrainer, df_train,  how='inner', left_on='ogr_DogID', right_on='dog_DogID')
#del merge_puppy_and_puptrainer['dog_DogID']


# In[53]:

del final_merge['dog_DogID']


# In[ ]:




# In[54]:

final_merge.head(5)



# In[55]:

del final_merge['ogr_DogID']


# In[56]:

final_merge.head(5)


# In[57]:

#merge_puppy_and_puptrainer.to_csv('Puppy_Trainer_TextandNumeric.csv', sep=',', index=None)


# In[58]:

target_for_values_Lab = {'Labrador Retriever': [
'Laab',
'Lab','lab','Yellow labrador','yellow labrador retriever','Yellow Labrador',
'Lab mix',
'Lab Ret',
'Lab retreiver',
'Lab retriever',
'Lab Retriver',
'LAB RTVR',
'Lab.',
'Lab. Ret.',
'Lab. Retriever',
'Labador',
'Labador Retreiver',
'Labadore',
'Labardor',
'Laborador',
'Laborador Retreiver',
'Laborador Retriever',
'Laborator Retriever',
'Labordor','YELLOW LAB',
'Labrabor Retriever',
'Labrabor Retriver',
'Labrado','Labrado retriever',
'Labrado Retriever',
'Labrador','LABRADOR','Labrador Retriver','labrador retreiver',
'Labrador  Retriever','labrodor retriever',
'Labrador Regtiever',
'LAbrador Ret',
'Labrador Retiever',
'Labrador Retreiver',
'Labrador Retrever',
'Labrador Retrieer',
'Labrador Retrieve',
'Labrador retrievef',
'Labrador Retriever',
'Labrador Retrievor',
'Labrador retriver',
'Labrador Rtvr',
'Labradore','Labrador retriever',       
'Labradore Retriever',
'Labradort',
'Labrardor retriever',
'Labrator Retreiver',
'Labrdor Retriever',
'Labrodor Retreiver',
'Labrodor Retriever',
'Labrodour Retriever',
'Laby',
'Ladbrador','L','laborador retriever',
'Lan', 'LR',
'Larador Retriever',
'Larbrador','Labrador Retrever',
'Larbrdor retriever',
'LBM','laab',
'lab', 'LR','Retriever, Labrador','Yellow Lab',
'yellow labrador','yellow lab',
'Yellow Labrador Retriever',
'lab mix',
'lab Ret','lab ret',
'lab retreiver',
'lab retriever',
'lab Retriver',
'lAB RTVR',
'lab.',
'lab. Ret.',
'lab. Retriever',
'labador',
'labador Retreiver',
'labadore',
'labardor',
'laborador',
'laborador Retreiver',
'laborador Retriever',
'laborator Retriever',
'labordor',
'labrabor Retriever',
'labrabor Retriver',
'labrado',
'labrado Retriever',
'labrador',
'labrador  Retriever',
'labrador Regtiever',
'lAbrador Ret',
'labrador Retiever',
'labrador Retreiver',
'labrador Retrever',
'labrador Retrieer',
'labrador Retrieve',
'labrador retrievef',
'labrador Retriever',
'labrador Retrievor',
'labrador retriver','laborator retriever',
'labrador Rtvr',
'labradore',
'labradore Retriever',
'labradort',
'labrardor retriever',
'labrator Retreiver',
'labrdor Retriever',
'labrodor Retreiver',
'labrodor Retriever',
'labrodour Retriever','Lab Retreiver','labrado retriever',
'laby','LAB RETRIEVER','Lab Retriever','Lab Rtvr',
'ladbrador','LY M','LYM','lym',
'lan',
'larador Retriever',
'larbrador','labrador retriever',
'larbrdor retriever','LAb','lab',
'BL',
'bl. lab','LAB','LAB','L','LaB','Lab',
'black','lab',
'black lab','yellow lab','Yellow lab','yellow Lab',
'Black Labrador','Black Lab','Black lab',
'Black Labrador Retriever','Blk Lab',
'blk lab','Babrador Retriever','English Labrador Retriever',
'American Lab','Brindle Lab',' lab',' Lab',' LAB',' laborador',' Labrador',' Laborador Retriever'
    ]}

for k, v in target_for_values_Lab.iteritems(): 
      for word in v:
        #print word
        #print " " + k
           #final_merge['GeneralComments'] = final_merge.apply(clean_text)
        final_merge['Breed']=final_merge.loc[:,'Breed'].replace(word,k)

target_for_values_GS = {
   'German Shepherd' : ['German Shepherd','G.S.','German Sheperd', 'Geman Shepherd','German Shep','Shepard','Belgen Sharpart','Shepherd',
                        'German shepard','German shepherd','german shepherd','german shepard',
                        'german shepard', 'German Sheperd','German Shephard', 'german shepherd dog',
                        'German Shepher','German Shepherd Dog','german shepherd dog','GSD',
                        'Greman Shepherd', 'GS','Greman Shepherd','belgan sharpart','German Shepherd',
                        'gsd','GSD','gsp','Lovely  German Sharpard','German Shepard','German Sheperd',
                        'Shepherd', 'GSD','GED', 'GDD', 'GDS','GSP','German Shepard','German Shepherd']}

    
    
for k, v in target_for_values_GS.iteritems(): 
      for word in v:
        #print word
        #print " " + k
        final_merge['Breed']=final_merge.loc[:,'Breed'].replace([word],[k])

        

target_for_values_LG = {
    'Labrador/Golden Cross': ['Labrador/Golden Cross','GLAB','Glab','GLab','glab','GLABYM','Golden Lab',
                             'golden lab','golden lab','Golden lab']}


for k, v in target_for_values_LG.iteritems():
#     #df1.loc[df.Breed.isin(v), 'Breed'] = k
#     df1.loc[:,'Breed'].replace(v,k,inplace=True)
    for word in v:
        #print word
        #print "    " +k
        final_merge['Breed']=final_merge.loc[:,'Breed'].replace(word,k)

target_for_values_Gol = {'Golden Retriever': ['Golden Retriever','Golden', 'Golden Retreiver',
'Golden retriever','golden','golden retriever',
'GR',
'Golden',
'Noble']}
for k, v in target_for_values_Gol.iteritems(): 
      for word in v:
        #print word
        #print " " + k
           final_merge['Breed']= final_merge.loc[:,'Breed'].replace(word,k)

target_for_values = {
    'F': ['bitch','remale','f','Female','fem','fema;e','Femae','Femail','Femaile','Femal','Femal3','Female','Femalw','femle','FEMALE','female','Girl','ID#2099','girl','n/a','None','own','Unknown','1364 & 655','1112/1329','065 102 601','2052','2235','11796','1972','1677','1649','1590','1395','1070','219','0','696','1018','ID# 2099','femal','femalw','Famale','femaile','femail']}

for k, v in target_for_values.iteritems():
    final_merge.loc[final_merge.Sex.isin(v), 'Sex'] = k
    


for k, v in target_for_values.iteritems():
      final_merge.loc[ final_merge.Sex.isin(v), 'Sex'] = k

target_for_values = {
    'M': ['Male',' Neutered Male','1110','1231','1627','1644','1766','1870','2019','??','1JJ11','boy','Crate from Val and Jim Hazlin','don\'t have one.','M - neutered','maie','Mail','Maile','Make','make','Male - neutered','male (neutered)','"Male, neutered"','Male1832','mine doesn\'t have a number?','N/A','NA','Neutered Male','new crate','none','own crate','Weren\'t given a crate','m','male','MALE','Male', 'Male','neutered mail','mail','Male, neutered']}

for k, v in target_for_values.iteritems():
    final_merge.loc[final_merge.Sex.isin(v), 'Sex'] = k
for k, v in target_for_values.iteritems():
       final_merge.loc[ final_merge.Sex.isin(v), 'Sex'] = k
  

#Change to Black:

target_for_values = {
    'Black': ['B','Bl','Bl','blac','black (and beautiful)','blck,','Blk','Blk.','blsck','back','black','BLACK','blk','BLK','color','lab','Back','blck','BLK.','Color','Black']}

for k, v in target_for_values.iteritems():
    final_merge.loc[final_merge.Color.isin(v), 'Color'] = k
    

#Change to Black/tan

target_for_values = {
    'Black/tan': ['Black/tan','B & T','b/t','B&T','B+T','bl and tan','Black & Tan','Black &tan','Black + Tan','Black and ran','Black and tan','Black and tan (?)','Black Brown','black tan','black w/ tan','Black, tan','Black, tan, silver','Black,tan','Black/ Tan','black/brown','Black/Tan','black+ tan','Blk & Tan','Blk and Tan','Blk/Tan','Brown & Black','brown black','Brown-Black','Brown, black','Brown/Black','Brown/Black/Tan','Coated Black','tan and black','Tan/Black','black and tan','Black and Tan','Bicolor (Black & red)','Bicolor (black w/ brown legs)','Black & red','Black and ran','Black and Red','black and white','Tri','Tri color','Brindle','GSD','B/T','b&t','Black / Tan','black & tan','black, tan','Black/tan','blk and tan','Blk/TAn','tan/black','Blk and tan','Black & tan']}

for k, v in target_for_values.iteritems():
    final_merge.loc[final_merge.Color.isin(v), 'Color'] = k
    
#Change to Yellow

target_for_values = {
   'Yellow' : ['Yellow','blond','yellow','Blond/Yellow','Blonde','blondelab lode','Butterscotch','Carmel Yellow','cream','Cream','darkish brown','fox red','Gold','gold','golden','GOLDEN','Golden Yellow','Lab','light tan','light yellow','Light Yellow','red','Red','Red Fox','Rust','Tan','tan','WELLOW','Wheat','white','White','White and yellow','White/Yellow','Y','y','yel','Yel','Yellllow','Yello','YELLO','yello','yelloiw','Yellow','yellow','YELLOW','Yellow - Dark','Yellow (red)','Yellow & White','yellow lab','Yellow with black trim','Yellow/Butterscotch','yellow/cream','Yellow/White','yellow1','yellowf','Light yellow','Yellowf']}
    
for k, v in target_for_values.iteritems():
    final_merge.loc[final_merge.Color.isin(v), 'Color'] = k    

#Change to Golden

target_for_values = {
   'Golden' : ['Golden','camel','golden/red','goldish','honey','Light Golden','Medium Gold','red/gold','reddish gold','warm gold','warm honey','Tan/Gold']}

for k, v in target_for_values.iteritems():
    final_merge.loc[final_merge.Color.isin(v), 'Color'] = k

#Change to Sable
target_for_values = {
   'Sable'  : ['Sable','Coated Sable','Sable','sable']}
 
for k, v in target_for_values.iteritems():
    final_merge.loc[final_merge.Color.isin(v), 'Color'] = k 
    
target_for_values = {
  'Yes'  : ['Yes']}
 
for k, v in target_for_values.iteritems():
    final_merge.loc[final_merge.GoodAppetite.isin(v), 'GoodAppetite'] = k  
    
target_for_values = {
  'No'  : ['No']}
 
for k, v in target_for_values.iteritems():
    final_merge.loc[final_merge.GoodAppetite.isin(v), 'GoodAppetite'] = k

target_for_values_RaiserState = {
   'CT' : ['CT','Connecticut','Ct',' CT','CT`']}

for k, v in target_for_values_RaiserState.iteritems():
    for word in v:
        final_merge['RaiserState']= final_merge.loc[:,'RaiserState'].replace(word,k)
        
target_for_values_RaiserState = {
   'CO' : ['CO']}
for k, v in target_for_values_RaiserState.iteritems():
    for word in v:
        final_merge['RaiserState']= final_merge.loc[:,'RaiserState'].replace(word,k)

target_for_values_RaiserState = {
   'DC' : ['DC']}
for k, v in target_for_values_RaiserState.iteritems():
    for word in v:
        final_merge['RaiserState']= final_merge.loc[:,'RaiserState'].replace(word,k)
        
target_for_values_RaiserState = {
    'DE': ['DE','Delaware']}
for k, v in target_for_values_RaiserState.iteritems():
    for word in v:
        final_merge['RaiserState']= final_merge.loc[:,'RaiserState'].replace(word,k)
        
target_for_values_RaiserState = {
    'DE': ['DE','Delaware']}
for k, v in target_for_values_RaiserState.iteritems():
    for word in v:
        final_merge['RaiserState']= final_merge.loc[:,'RaiserState'].replace(word,k)

target_for_values_RaiserState = {
    'IN': ['IN']}
for k, v in target_for_values_RaiserState.iteritems():
    for word in v:
        final_merge['RaiserState']= final_merge.loc[:,'RaiserState'].replace(word,k)
        
        
target_for_values_RaiserState = {
   'ME' : ['ME','maine','Maine']}
for k, v in target_for_values_RaiserState.iteritems():
    for word in v:
        final_merge['RaiserState']= final_merge.loc[:,'RaiserState'].replace(word,k)
    
        
target_for_values_RaiserState = {
  'MD'  : ['MD','Md','Maryland']}
for k, v in target_for_values_RaiserState.iteritems():
    for word in v:
        final_merge['RaiserState']= final_merge.loc[:,'RaiserState'].replace(word,k)
        
target_for_values_RaiserState = {
   'MA' : ['MA','Massachusetts','Ma',' Ma']}
for k, v in target_for_values_RaiserState.iteritems():
    for word in v:
        final_merge['RaiserState']= final_merge.loc[:,'RaiserState'].replace(word,k)
        

target_for_values_RaiserState = {
   'MD' : ['MD','Md.']}
for k, v in target_for_values_RaiserState.iteritems():
    for word in v:
        final_merge['RaiserState']= final_merge.loc[:,'RaiserState'].replace(word,k)

target_for_values_RaiserState = {
   'NC' : ['NC','N.C.']}
for k, v in target_for_values_RaiserState.iteritems():
    for word in v:
        final_merge['RaiserState']= final_merge.loc[:,'RaiserState'].replace(word,k)
target_for_values_RaiserState = {
   'NH' : ['New Hampshire','NH']}
for k, v in target_for_values_RaiserState.iteritems():
    for word in v:
        final_merge['RaiserState']= final_merge.loc[:,'RaiserState'].replace(word,k)



target_for_values_RaiserState = {
   'WV' : ['WV','wv']}
for k, v in target_for_values_RaiserState.iteritems():
    for word in v:
        final_merge['RaiserState']= final_merge.loc[:,'RaiserState'].replace(word,k)


target_for_values_RaiserState = {
   'NY' : ['NY','New  York','N.Y.','New York','New YHork','Ny']}
for k, v in target_for_values_RaiserState.iteritems():
    for word in v:
        final_merge['RaiserState']= final_merge.loc[:,'RaiserState'].replace(word,k)
        
target_for_values_RaiserState = {
   'OH' : ['OH','Ohio',' OH','Ohio','ohio','OH','Oh']}
for k, v in target_for_values_RaiserState.iteritems():
    for word in v:
        final_merge['RaiserState']= final_merge.loc[:,'RaiserState'].replace(word,k)

target_for_values_RaiserState = {
  'PA'  : ['PA','Pennsylvania']}
for k, v in target_for_values_RaiserState.iteritems():
    for word in v:
        final_merge['RaiserState']= final_merge.loc[:,'RaiserState'].replace(word,k)
    
target_for_values_RaiserState = {
  'VA'  : ['va','VA','VA.','V','\'VA','Virginia','\'VIRGINIA','\'Virginia','\'virginia',' VA']}
for k, v in target_for_values_RaiserState.iteritems():
    for word in v:
        final_merge['RaiserState']= final_merge.loc[:,'RaiserState'].replace(word,k)

target_for_values_RaiserState = {
   'VI' : ['VI','Virgin Islands']}
for k, v in target_for_values_RaiserState.iteritems():
    for word in v:
        final_merge['RaiserState']= final_merge.loc[:,'RaiserState'].replace(word,k)
    
target_for_values_RaiserState = {
  'VT'  : ['VT','\'Vermont']}
for k, v in target_for_values_RaiserState.iteritems():
    for word in v:
        final_merge['RaiserState']= final_merge.loc[:,'RaiserState'].replace(word,k)


# In[59]:

final_merge.to_csv('Puppy_Trainer_TextandNumeric.csv', sep=',', index=None)


# In[ ]:



