"""
To run this file use the following command in terminal:
python combined_script.py --inp Westmoreland --out WestmorelandResults \
--name_map Westmoreland_Judges_Name_Map.csv 

Or if you multiple python versions, specify which one:
python3 run_county_analysis.py --inp Westmoreland --out WestmorelandResults \
--name_map Westmoreland_Judges_Name_Map.csv 

Here:
- inp denotes the Input County file
- out denotes the Output Folder where the resuls will be stored
- name_map denotes the Names mapping file for Judges of a particular county
- gravity_map denotes the path to Gravity score mapping

Note:
1)If you want to specify a different path for gravity_map, specify following:
--gravity_map offense_gravity_map.csv
2)If you want to turn off the output from the terminal use:
--logs False

"""
import argparse
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import sys

import warnings
warnings.filterwarnings("ignore")

# %matplotlib inline
pd.options.display.float_format = '{:20,.2f}'.format
pd.set_option('display.max_columns',500)

parser = argparse.ArgumentParser(description='Running the combined script...')
parser.add_argument("--inp", required=True, type=str, help="The relative path to County data")
parser.add_argument("--name_map", required=True, type=str, help="Mapping for Judges with multiple names matches")
parser.add_argument("--gravity_map", default="offense_gravity_map.csv", type=str, \
    help="Mapping for Title,Section,SubSection to Gravity values")
parser.add_argument("--out", required=True, type=str, help="Output folder for storing results")
parser.add_argument("--logs", default=True, type=bool, help="Output folder for storing results")

args = parser.parse_args()
input_file = args.inp
gravity_map_file = args.gravity_map
output_folder = args.out
name_map = args.name_map


assert os.path.exists(gravity_map_file), "Gravity Map file not found"
assert os.path.exists(input_file), "Input County file not found"

if not os.path.exists(output_folder):
  os.mkdir(output_folder)


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(os.path.join(output_folder,"logfile.txt"), "w")

    def write(self, message):
        if args.logs:
          self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass

sys.stdout = Logger()

# input_file = "WestmorelandData.csv" #comment this later

df = pd.read_csv(input_file, sep='\t', encoding='ISO-8859-1')
print("Reading County data...")
df = df.drop_duplicates()

name_map_exists = False

try:
    name_df = pd.read_csv(name_map)
    nameMap = {}
    for index, row in name_df.iterrows():
      nameMap[row['Original_Name']] = row['New_Name']

    df.BailActionAuthority = df.BailActionAuthority.apply(lambda x : nameMap[x] if x in nameMap else x)

    name_map_exists = True
except:
    print("No name map found for {}".format(input_file))

print("Current dataset has {} rows".format(len(df)))

df.BailActionDate = pd.to_datetime(df.BailActionDate)
df['hasBail'] = df.BailActionAmount.apply(lambda x : False if x==0.0 or pd.isna(x) else True)

# number of cases where there were more than 1 bail amount associated with the same OTN
multiple_bail_count = sum(df.groupby(['OTN']).agg(lambda df: len(df['BailActionAmount'].unique()) > 1)['BailActionAmount'])
print("{} OTNs have more than 1 bail amount associated with them.".format(multiple_bail_count))

# Top 7 OTN count
df = df.sort_values(["BailActionDate", "System"], ascending = [True, False])
otn_counts_df = df['OTN'].value_counts().reset_index()
otn_counts_df.columns = ['OTN', 'Count']
otn_counts_df.head(7).to_csv(os.path.join(output_folder,'otn_counts_df.csv'),index=False)
otn_counts_df.head(7)
print("Writing top 7 OTN counts to otn_counts_df.csv")

# Getting Gravity Scores for OTN
print("")
print("Merging with the gravity map...")
gravity_map = pd.read_csv(gravity_map_file)

from pandas.api.types import is_numeric_dtype

# combine Title section subsection
df['SubSection'] = df['SubSection'].fillna("")
df['gravity key'] = df['Section'] + df['SubSection']
df['gravity key'] = df['gravity key'].str.replace("*", "").str.replace(".", "").str.upper()
if is_numeric_dtype(df['Title']):
	df['Title'] = df['Title'].apply(lambda x : x if pd.isna(x) else str(int(x)) )
df['gravity key'] = df["Title"].astype(str) + '.' + df['gravity key']

# merge with the map
df = df.merge(gravity_map, left_on = "gravity key", right_on = "Charge", how = "left")
df["MeanCleanedGravityScore"] = df["MeanCleanedGravityScore"].apply(lambda x: np.round(x).astype(int) \
                                                                                if pd.notnull(x) else x)
match_rate = (len(df[~df["MeanCleanedGravityScore"].isnull()]) / len(df)) * 100.0
print("Match rate is {0:.2f}%".format(match_rate))

df_otn = df.sort_values(by=['BailActionDate', 'System', 'MeanCleanedGravityScore'], ascending=[True,False,False])
df_otn = df_otn.drop_duplicates(subset = ['OTN'])
df_otn.to_csv(os.path.join(output_folder,'unique_otn_data.csv'), index=False)
print("After subsetting for unique OTN values, we have: {} rows".format(len(df_otn)))

print("")
print("Starting with Bail Amount analysis...")
# Bail Amount
df_hasBail = df_otn[df_otn["hasBail"] == True]
df_hasBail = df_hasBail[df_hasBail['BailType'] == 'Monetary']

# cases in 2017
# len(df_otn[(df_otn["BailActionDate"] > pd.to_datetime('1/1/2017')) & (df_otn["BailActionDate"] < pd.to_datetime('1/1/2018'))])

# first_date_for_each_otn = df.sort_values(by=['BailActionDate'], ascending=True).groupby("OTN").first()
# first_date_for_each_otn[first_date_for_each_otn["BailActionDate"] > pd.to_datetime('1/1/2018')]

bail_quantile = df_hasBail["BailActionAmount"].quantile(np.array(range(0, 105, 5))*0.01)
bail_quantile = pd.DataFrame(bail_quantile).reset_index()
bail_quantile.columns = ['Percentile', 'Bail Amount']
bail_quantile.to_csv(os.path.join(output_folder,'bail_quantile.csv'), index=False)
# print(bail_quantile)
print("Writing Distribution of Bail Amounts to bail_quantile.csv")

df_hasBail["Month"] = df_hasBail["BailActionDate"].dt.to_period('M')
bail_by_month = df_hasBail.groupby(["Month"])['BailActionAmount'].agg(['sum', 'count', 'mean']).reset_index()
bail_by_month.columns = ["Month","CumulativeBailAmount", "CountOf", "AverageBail"]
bail_by_month.to_csv(os.path.join(output_folder,'bail_by_month.csv'), index=False)
# print(bail_by_month)
print("Writing Bail Amounts per month to bail_by_month.csv")

plt.plot(bail_by_month.Month.dt.to_timestamp(), bail_by_month.AverageBail)

bail_by_month_subset = bail_by_month[bail_by_month["Month"].dt.to_timestamp() > pd.to_datetime('1/1/2015')]
bail_by_month_subset.head()

plt.plot(bail_by_month_subset.Month.dt.to_timestamp(), bail_by_month_subset.AverageBail)
plt.title('Avg Bail')
plt.savefig(os.path.join(output_folder,'avg_bail_by_month.png'))
print("Saved Average Bail curve as avg_bail_by_month.png")

"""# Bail Type"""
print("")
print("Starting Bail Type analysis...")
bail_type = df_otn.groupby(['BailType'])['OTN'].count().reset_index()
bail_type.columns = ["Bail Type", "Case Count"]
bail_type['Ratio'] = bail_type['Case Count']/len(df_otn)
bail_type['Ratio'] = bail_type['Ratio'].apply(lambda x: "{0:.2f}%".format(x*100))
bail_type.to_csv(os.path.join(output_folder,'bail_type.csv'),index=False)
print("Writing distribution of Bail Types to bail_type.csv")
# bail_type

# Average Bail Amt by Bail Type
avg_bail_type = df_otn.groupby(['BailType'])['BailActionAmount',].mean().reset_index()
avg_bail_type.columns = ["Bail Type", "Bail Amt"]
avg_bail_type.to_csv(os.path.join(output_folder,'avg_bail_by_type.csv'), index=False)
print("Writing Average Bail Amount by Bail Type to avg_bail_by_type.csv")

"""# Judges"""
print("")
print("Starting with Judges analysis...")
def by_judge(df):
    result = pd.Series()
    n = len(df)
    result['% Male Defendants'] = sum(df['DefendantSex'] == 'Male')/n
    result['% Female Defendants'] = sum(df['DefendantSex'] == 'Female')/n

    result['% Black Defendants'] = sum(df['DefendantRace'] == 'Black')/n
    result['% White Defendants'] = sum(df['DefendantRace'] == 'White')/n

    result['% cases Unsecured Bail set'] = sum(df['BailType'] =='Unsecured')/n
    result['% cases ROR Bail set'] = sum(df['BailType'] =='ROR')/n
    result['% cases Monetary Bail set'] = sum(df['BailType'] =='Monetary')/n
    result['% cases Nonmonetary Bail set'] = sum(df['BailType'] =='Nonmonetary')/n
    
    result['% cases (Unsecured + Monetary)'] = sum(df['BailType'] =='Unsecured')/n + sum(df['BailType'] =='Monetary')/n
    result['% cases (ROR + Nonmonetary)'] = sum(df['BailType'] =='ROR')/n + sum(df['BailType'] =='Nonmonetary')/n

    #result['% cases Public Defender Issued'] = sum(df[RepresentationType =='Public Defender'])/n
    #result['% cases Private Defender Issued'] = sum(df[RepresentationType =='Private'])/n

    result['Total Cases'] = n
    
    return result

if name_map_exists == True:
    df_otn['BailActionAuthority'] = df_otn['BailActionAuthority'].replace(nameMap)

judge = df_otn.groupby(['BailActionAuthority']).apply(by_judge).reset_index()
judge = judge.sort_values('% cases Monetary Bail set', ascending = False)
judge.loc[:,'% Male Defendants':'% cases (ROR + Nonmonetary)'] = judge.loc[:,'% Male Defendants':'% cases (ROR + Nonmonetary)'].applymap(lambda x: "{0:.2f}%".format(x*100))
judge.to_csv(os.path.join(output_folder,'judge.csv'),index=False)
print("Writing Judge analysis results to judge.csv")
# judge

"""# Offense Type"""
print("")
print("Starting with Offense Type analysis...")
chargeList = ['Use/Poss Of Drug Paraph', 'Int Poss Contr Subst By Per Not Reg','Poss Of Marijuana','Marijuana-Small Amt Personal Use',
              'Manufacture, Delivery, or Possession With Intent to Manufacture or Deliver','Simple Assualt',
            'Retail Theft-Take Mdse','Aggravated Assault']

out = df_otn[(df_otn.Description.isin(chargeList))]

df_offense = out.groupby('Description')['BailActionAmount'].agg(['count','mean']).reset_index()
df_offense.columns = ['Description','Number of Cases','Average Bail Amount']
df_offense.to_csv(os.path.join(output_folder,"offense_distrib_count_bailamount.csv"),index=False, float_format='%.2f')
# df_offense
print("Writing Distribution of Offenses and Average Bail to offense_distrib_count_bailamount.csv")

df_offense = out.groupby(['Description','BailType'])['BailActionAmount'].agg('mean').reset_index()
df_offense.columns = ['Description', 'BailType', 'mean']
df_offense = df_offense.pivot(index = 'Description', columns = 'BailType', values = 'mean').fillna(0)
df_offense.to_csv(os.path.join(output_folder,"offense_bailtype_means.csv"), float_format='%.2f')
# df_offense
print("Writing Average Bail Amounts by Offense Type and Bail Type to offense_bailtype_means.csv")

df_offense = out.groupby(['Description','BailType'])['BailActionAmount'].agg('count').reset_index()
df_offense.columns = ['Description', 'BailType', 'count']
df_offense = df_offense.pivot(index = 'Description', columns = 'BailType', values = 'count').fillna(0)
df_offense.to_csv(os.path.join(output_folder,"offense_bailtype_counts.csv"), float_format='%.2f')
# df_offense
print("Writing Number of cases by Offense Type and Bail Type to offense_bailtype_counts.csv")





"""# Offense Score"""
print("")
print("Starting with Offense Score analysis...")
df_hasScore = df[~df["MeanCleanedGravityScore"].isnull()]
print("Number of OTNs with Gravity score {}".format(len(df_hasScore)))
df_hasScore["MeanCleanedGravityScore"] = df_hasScore["MeanCleanedGravityScore"].astype(int)

case_bail_by_grade = df_hasScore.groupby("MeanCleanedGravityScore")["BailActionAmount"].agg(["mean","count"]).reset_index()
case_bail_by_grade["Proportion of Cases"] = case_bail_by_grade["count"] / sum(case_bail_by_grade["count"])
case_bail_by_grade["Proportion of Cases"] = case_bail_by_grade["Proportion of Cases"].apply(lambda x: "{0:.2f}%".format(x*100))
case_bail_by_grade.to_csv(os.path.join(output_folder,'case_bail_by_grade.csv'),index=False)
print("Writing Distribution of Cases and Average Bail by Offense Score to case_bail_by_grade.csv")
# case_bail_by_grade

bail_by_type_grade = df_hasScore.pivot_table(index=["MeanCleanedGravityScore"],columns=['BailType'], values='BailActionAmount', aggfunc=np.mean)
bail_by_type_grade = bail_by_type_grade.fillna(0)
bail_by_type_grade.to_csv(os.path.join(output_folder,'bail_by_type_grade.csv'))
print("Writing Average Bail by Bail Type and Offense Score to bail_by_type_grade.csv")
# bail_by_type_grade

case_by_type_grade = df_hasScore.pivot_table(index=["MeanCleanedGravityScore"],columns=['BailType'], values='BailActionAmount', aggfunc = len)
case_by_type_grade = case_by_type_grade.fillna(0)
case_by_type_grade.to_csv(os.path.join(output_folder,'case_by_type_grade.csv'))
print("Writing Number of Cases by Bail Type and Offense Score to case_by_type_grade.csv")
# case_by_type_grade

"""# Gender"""
print("")
print("Starting with Gender analysis...")
df_gender = df_otn.DefendantSex.value_counts().reset_index()
df_gender.columns = ['DefendantSex','Count']
df_gender.to_csv(os.path.join(output_folder,"gender_counts.csv"), index = False)
# df_gender
print("Writing Counts of Defendants by Gender to gender_counts.csv")

df_gender = df_otn.groupby(['DefendantSex'])['BailActionAmount'].agg('mean').reset_index()
df_gender.columns = ['DefendantSex','BailActionAmount']
df_gender.to_csv(os.path.join(output_folder,"gender_bailamount_mean.csv"), index = False, float_format='%.2f')
print("Writing Avg. Bail Amount by Gender to gender_bailamount_mean.csv")
# df_gender

df_gender = df_otn.groupby(['DefendantSex','BailType'])['BailActionAmount'].agg('count').reset_index()
df_gender.columns = ['DefendantSex','BailType','count']
df_gender = df_gender.pivot(index = 'DefendantSex', columns = 'BailType', values = 'count').fillna(0)
df_gender.to_csv(os.path.join(output_folder,"gender_bailtype_counts.csv"))
print("Writing Number of Cases by Bail Type and Gender to gender_bailtype_counts.csv")
# df_gender

df_gender = df_otn.groupby(['DefendantSex','BailType'])['BailActionAmount'].agg('mean').reset_index()
df_gender.columns = ['DefendantSex','BailType','mean']
df_gender = df_gender.pivot(index = 'DefendantSex', columns = 'BailType', values = 'mean').fillna(0)
df_gender.to_csv(os.path.join(output_folder,"gender_bailtype_mean.csv"), float_format='%.2f')
print("Writing Avg. Bail Amount by Bail Type and Gender to gender_bailtype_mean.csv")
# df_gender

female = df_otn[df_otn.DefendantSex=='Female']
male = df_otn[df_otn.DefendantSex=='Male']

f_vc = (female.Description.value_counts()/len(female)).reset_index()
m_vc = (male.Description.value_counts()/len(male)).reset_index()

f_vc.columns = ['Description','Counts']
m_vc.columns = ['Description','Counts']

diffList = []

for case in m_vc.Description.unique():
    
    if case in f_vc.Description.unique():
        
        male_ratio = list(m_vc[m_vc.Description==case]['Counts'])[0]
        female_ratio = list(f_vc[f_vc.Description==case]['Counts'])[0]
        
        diffList.append((case, male_ratio - female_ratio))

diffList = sorted(diffList, key = lambda x : x[1])

#diffList

## Crimes commited more by women than men
diffList[0:5]

## Crimes commited more by men than women
diffList[-5:]

women = [x[0] for x in diffList[0:5]]
men = [x[0] for x in diffList[-1:-6:-1]]
# display(women)
# display(men)

gender_crimes_comparison = pd.DataFrame({'Crimes more commonly committed by Women': women, 'Crimes more commonly committed by Men': men})
gender_crimes_comparison.to_csv(os.path.join(output_folder,"gender_crimes_comparison.csv"), index=False)
print("Writing Differences in nature of crimes committed by the 2 genders to gender_crimes_comparison.csv")
# gender_crimes_comparison


df_gravity = df_otn.groupby(['MeanCleanedGravityScore','DefendantSex'])['BailActionAmount'].agg('mean').reset_index()

df_gravity.columns = ['Gravity Score','DefendantSex','BailActionAmount']
print(df_gravity.shape)
df_gravity = df_gravity.pivot(index = 'Gravity Score', columns = 'DefendantSex', values = 'BailActionAmount').fillna(0)
print(df_gravity.shape)

if 'Unreported/Unknown' in df_gravity.columns:
    df_gravity = df_gravity.drop(columns=['Unreported/Unknown'])

df_gravity.columns = ['BailAmount_Female','BailAmount_Male']
df_gravity['Ratio'] = df_gravity['BailAmount_Male']/df_gravity['BailAmount_Female']
df_gravity.to_csv(os.path.join(output_folder,"gravity_gender_analysis.csv"), float_format='%.2f')
print("Writing Comparison of Avg Bail amount per offense gravity across Genders to gravity_gender_analysis.csv")
# df_gravity

df_gravity_female = df_otn[df_otn['DefendantSex'] == 'Female']
female_ct = len(df_gravity_female)
df_gravity_female = df_gravity_female.groupby('MeanCleanedGravityScore').apply(lambda x: len(x.index)/female_ct*100).reset_index()
df_gravity_female['DefendantSex'] = 'Female'

df_gravity_male = df_otn[df_otn['DefendantSex'] == 'Male']
male_ct = len(df_gravity_male)
df_gravity_male = df_gravity_male.groupby('MeanCleanedGravityScore').apply(lambda x: len(x.index)/male_ct*100).reset_index()
df_gravity_male['DefendantSex'] = 'Male'

df_gravity = pd.concat([df_gravity_female, df_gravity_male], axis=0)
df_gravity.columns = ['Gravity Score', 'Percentage', 'DefendantSex']
df_gravity = df_gravity.pivot(index = 'Gravity Score', columns = 'DefendantSex', values = 'Percentage').fillna(0)
df_gravity.to_csv(os.path.join(output_folder,"gravity_gender_distribution_by_perc.csv"), float_format='%.2f')
print("Writing Proportion of Offense Gravoty by Gender to gravity_gender_distribution_by_perc.csv")
# df_gravity

"""# Race"""
print("")
print("Starting with Race analysis...")
# Counts of Defendants by Race
df_race_count = df_otn.groupby(['DefendantRace']).apply(lambda x : len(x.index))
df_race_count = df_race_count.sort_values(ascending=False).reset_index()
df_race_count.columns = ['DefendantRace', 'Count']
df_race_count.to_csv(os.path.join(output_folder,"race_counts.csv"), index = False)
print("Writing Counts of Defendants by Race to race_counts.csv")
# df_race_count

# Average Bail Amount by Race
df_avgbail_amt_race = df_otn.groupby(['DefendantRace'])[['BailActionAmount']].agg('mean')
df_avgbail_amt_race = df_avgbail_amt_race.sort_values(by='BailActionAmount', ascending=False).reset_index()
df_avgbail_amt_race.to_csv(os.path.join(output_folder,"race_bailamount_mean.csv"), index = False)
print("Writing Average Bail Amount by Race to race_bailamount_mean.csv")
# df_avgbail_amt_race

#Number of Cases by Bail Type and Race
df_bailtype_race = df_otn.groupby(['DefendantRace', 'BailType']).apply(lambda x : len(x.index)).reset_index()
df_bailtype_race = df_bailtype_race.rename(columns = {0: 'Count'})
df_bailtype_race = df_bailtype_race.pivot(index = 'DefendantRace', columns = 'BailType', values = 'Count').fillna(0)
df_bailtype_race.to_csv(os.path.join(output_folder,"race_bailtype_count.csv"), float_format='%.2f')
print("Writing Number of Cases by Bail Type and Race to race_bailtype_count.csv")
# df_bailtype_race

#Average Bail by Bail Type and Race
df_bailtypeavg_race = df_otn.groupby(['DefendantRace', 'BailType'])[['BailActionAmount']].agg('mean').reset_index()
df_bailtypeavg_race = df_bailtypeavg_race.pivot(index = 'DefendantRace', columns = 'BailType', values = 'BailActionAmount').fillna(0)
df_bailtypeavg_race.to_csv(os.path.join(output_folder,"race_bailtype_avg.csv"), float_format='%.2f')
print("Writing Average Bail by Bail Type and Race to race_bailtype_avg.csv")
# df_bailtypeavg_race



