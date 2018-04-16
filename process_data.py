import pandas as pd
import numpy as np
import csv


# load outcome data and generate perionId_dogId_pair as the key used in merge
outcome_df = pd.read_csv('./datasets/PuppyTrainerOutcome-after-step2.csv')
outcome_df['perionId_dogId_pair'] = outcome_df.ogr_PersonID.astype(str).str.cat(outcome_df.dog_DogID.astype(str), sep='-')
print(outcome_df.head(5))

# load trainer data
# perionId_dogId_pair is generated in csv file prior to loading to dataframe due to format issue when concatenate in code.
trainer_df = pd.read_csv('./datasets/TrainerInfo-after-step3.csv')
print(trainer_df.head(5))


# merge outcome and trainer information by perionId_dogId_pair
trainer_outcome_df = pd.merge(outcome_df, trainer_df,  how='inner', left_on='perionId_dogId_pair', right_on='perionId_dogId_pair')

# keep data consistent
# replace dog_SubStatusCode with label
trainer_outcome_df.loc[:,'dog_SubStatusCode'].replace([1,2,3,17,21,27,28,29,37,45,50,66,76,97,111,154,159,160,161,162,166,167,171,172,173,174,175,179,181,184,187,188],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],inplace=True)
trainer_outcome_df.loc[:,'dog_SubStatusCode'].replace([23,25,26,27,55,98,99,121,169],[1,1,1,1,1,1,1,1,1],inplace=True)
print(trainer_outcome_df.shape)
print(trainer_outcome_df.head(5))

# remove duplicates
clean_df = trainer_outcome_df.drop_duplicates(['DayInLife'], keep='first')
print(clean_df.shape)


# output to csv
clean_df.to_csv('./datasets/trainer_daylife_and_outcome.csv', index=None)



