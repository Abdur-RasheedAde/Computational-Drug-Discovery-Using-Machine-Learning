#!/usr/bin/env python
# coding: utf-8

# ## Computational Drug Discovery
# 
# New notebook

# ### **Computational Drug Discovery Download Bioactivity Data**
# 
# Abdur-Rasheed Adeoye
# 
# This is a **Bioinformatoics** real-life data science project particularly, building a machine learning model using the ChEMBL bioactivity data.

# In[42]:


#Import all relevant Libraries

get_ipython().system('pip install seaborn')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ## **ChEMBL Database**
# 
# The [*ChEMBL Database*](https://www.ebi.ac.uk/chembl/) is a database that contains curated bioactivity data of more than 2 million compounds. It is compiled from more than 76,000 documents, 1.2 million assays and the data spans 13,000 targets and 1,800 cells and 33,000 indications.
# [Data as of August 07, 2025; ChEMBL version 26].

# In[27]:


# Installing libraries
# Install the ChEMBL web service package so as to retrieve bioactivity data from the ChEMBL Database.

get_ipython().system(' pip install chembl_webresource_client')


# In[28]:


# Import necessary libraries
import pandas as pd
from chembl_webresource_client.new_client import new_client


# In[29]:


## Search for Target protein

# Target search for coronavirus
target = new_client.target
target_query = target.search('coronavirus')
targets = pd.DataFrame.from_dict(target_query)
targets


# ### **Select and retrieve bioactivity data for *SARS coronavirus 3C-like proteinase* (seventh entry)**
# 
# We will assign the seventh entry (which corresponds to the target protein, *coronavirus 3C-like proteinase*) to the ***selected_target*** variable 

# In[30]:


selected_target = targets.target_chembl_id[6]
selected_target


# In[31]:


#Retrieve only bioactivity data for coronavirus 3C-like proteinase (CHEMBL3927) that are reported as IC$_{50}$ values in nM (nanomolar) unit.
activity = new_client.activity
res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")

df = pd.DataFrame.from_dict(res)

df.head()


# In[32]:


#Save result of bioactivity data to a CSV file bioactivity_data.csv.

df.to_csv('/lakehouse/default/Files/bioactivity_data_raw.csv', index=False)


# In[33]:


import pandas as pd
# Load data into pandas DataFrame from "/lakehouse/default/Files/bioactivity_data_raw.csv"
df2 = pd.read_csv("/lakehouse/default/Files/bioactivity_data_raw.csv")
display(df2)


# In[34]:


df2.info()


# In[35]:


#drop any missing value in target column
df2 = df[df.standard_value.notna()]
df2.head()


# In[36]:


#Check the unique values in 'standard_type'
df2['standard_value'].unique()


# #### **Data Prprocessing of the bioactivity data**
# 
# ###### Labeling compounds as either being active, inactive or intermediate
# The bioactivity data is in the IC50 unit. Compounds having values of less than 1000 nM will be considered to be **active** while those greater than 10,000 nM will be considered to be **inactive**. As for those values in between 1,000 and 10,000 nM will be referred to as **intermediate**. 

# In[37]:


bioactivity_class = []
for i in df2.standard_value:
  if float(i) >= 10000:
    bioactivity_class.append("inactive")
  elif float(i) <= 1000:
    bioactivity_class.append("active")
  else:
    bioactivity_class.append("intermediate")


# In[38]:


print(len(bioactivity_class))


# In[39]:


#Select only 3 essential columns
selection = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']
df3 = df2[selection]
df3


# In[40]:


#append 'bioactivity_class' to Dataframe
bioactivity_class_series = pd.Series(bioactivity_class, name='bioactivity_class')

# Concatenate the DataFrame and the named Series
df4 = pd.concat([df3, bioactivity_class_series], axis=1)


# In[41]:


df4


# In[42]:


#Save data to csv 

df4.to_csv('/lakehouse/default/Files/bioactivity_preprocessed_data.csv', index=False)


# ## **Exploratory Data Analysis**

# In[1]:


# Install conda and rdkit

#rdkt: help in molecular representation, allow working with SMILES strings, molecular graphs, and 3D structures.

get_ipython().system(' wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh')
get_ipython().system(' chmod +x Miniconda3-py37_4.8.2-Linux-x86_64.sh')
get_ipython().system(' bash ./Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -f -p /usr/local')
get_ipython().system(' conda install -c rdkit rdkit -y')
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/')


# ##### **Calculate Lipinski descriptors**
# Christopher Lipinski, a scientist at Pfizer, came up with a set of rule-of-thumb for evaluating the **druglikeness** of compounds. Such druglikeness is based on the Absorption, Distribution, Metabolism and Excretion (ADME) that is also known as the pharmacokinetic profile. Lipinski analyzed all orally active FDA-approved drugs in the formulation of what is to be known as the **Rule-of-Five** or **Lipinski's Rule**.
# 
# The Lipinski's Rule stated the following:
# * Molecular weight < 500 Dalton
# * Octanol-water partition coefficient (LogP) < 5
# * Hydrogen bond donors < 5
# * Hydrogen bond acceptors < 10 

# In[2]:


#import all necessary libraries
import pandas as pd
import numpy as np


# In[3]:


# Run this cell to install RDKit in Colab
get_ipython().system('pip install rdkit-pypi')

# Then import RDKit
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


# In[4]:


#Calculate descriptors

# Inspired by: https://codeocean.com/explore/capsules?query=tag:data-curation

def lipinski(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem) 
        moldata.append(mol)
       
    baseData= np.arange(1,1)
    i=0  
    for mol in moldata:        
       
        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
           
        row = np.array([desc_MolWt,
                        desc_MolLogP,
                        desc_NumHDonors,
                        desc_NumHAcceptors])   
    
        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1      
    
    columnNames=["MW","LogP","NumHDonors","NumHAcceptors"]   
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)
    
    return descriptors


# In[5]:


import pandas as pd
# Load data into pandas DataFrame from "/lakehouse/default/Files/bioactivity_preprocessed_data.csv"
df = pd.read_csv("/lakehouse/default/Files/bioactivity_preprocessed_data.csv")
df.head()


# In[6]:


# Step 1: Replace NaNs or floats with empty strings or valid SMILES
df['canonical_smiles'] = df['canonical_smiles'].fillna('').astype(str)

# Step 2: Call the lipinski function
df_lipinski = lipinski(df['canonical_smiles'])

# Step 3: Display or use the result
print(df_lipinski)


# In[7]:


#create a variable
df_lipinski = lipinski(df.canonical_smiles)

df_lipinski


# In[8]:


#the bioactivity_preprocessed_data
df


# In[9]:


# Combine DataFrames df_lipinski and bioactivity_preprocessed_data

df_combined = pd.concat([df,df_lipinski], axis=1)

df_combined


# ### **Convert IC50 to pIC50**
# To allow **IC50** data to be more uniformly distributed, we will convert **IC50** to the negative logarithmic scale which is essentially **-log10(IC50)**.
# 
# This custom function pIC50() will accept a DataFrame as input and will:
# * Take the IC50 values from the ``standard_value`` column and converts it from nM to M by multiplying the value by 10$^{-9}$
# * Take the molar value and apply -log10
# * Delete the ``standard_value`` column and create a new ``pIC50`` column

# In[10]:


import numpy as np

def pIC50(input):
    pIC50 = []

    for i in input['standard_value_norm']:
        molar = i*(10**-9) # Converts nM to M
        pIC50.append(-np.log10(molar))

    input['pIC50'] = pIC50
    x = input.drop('standard_value_norm', axis=1)
        
    return x


# In[11]:


#Point to note: Values > 100,000,000 will be fixed at 100,000,000 otherwise the negative logarithmic value will become negative.

df_combined.standard_value.describe()


# In[12]:


-np.log10( (10**-9)* 100000000 )


# In[13]:


-np.log10( (10**-9)* 10000000000 )


# In[14]:


def norm_value(input):
    norm = []

    for i in input['standard_value']:
        if i > 100000000:
          i = 100000000
        norm.append(i)

    input['standard_value_norm'] = norm
    x = input.drop('standard_value', axis=1)
        
    return x


# In[15]:


#Applying the norm_value() function so that the values in the standard_value column is normalized.

df_norm = norm_value(df_combined)
df_norm


# In[16]:


df_norm.standard_value_norm.describe()


# In[17]:


df_final = pIC50(df_norm)
df_final


# In[18]:


df_final.pIC50.describe()


# In[19]:


### Removing the 'intermediate' bioactivity class

df_2class = df_final[df_final.bioactivity_class != 'intermediate']
df_2class


# ##### **Exploratory Data Analysis (Chemical Space Analysis) via Lipinski descriptors**

# In[20]:


get_ipython().system('pip install seaborn')


# In[21]:


#Import important libraries
import seaborn as sns
sns.set(style='ticks')
import matplotlib.pyplot as plt


# ### **Frequency plot of the 2 bioactivity classes**

# In[30]:


# Scatter plot of MW versus LogP

#It can be seen that the 2 bioactivity classes are spanning similar chemical spaces as evident by the scatter plot of MW vs LogP.

plt.figure(figsize=(5.5, 5.5))

sns.scatterplot(x='MW', y='LogP', data=df_2class, hue='bioactivity_class', size='pIC50', edgecolor='black', alpha=0.7)

plt.xlabel('MW', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.savefig('/lakehouse/default/Files/plot_MW_vs_LogP.pdf')


# In[31]:


# Box plots

plt.figure(figsize=(5.5, 5.5))

sns.boxplot(x = 'bioactivity_class', y = 'pIC50', data = df_2class)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('pIC50 value', fontsize=14, fontweight='bold')

plt.savefig('/lakehouse/default/Files/plot_ic50.pdf')


# In[24]:


#Statistical analysis | Mann-Whitney U Test

def mannwhitney(descriptor, verbose=False):
  # https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
  from numpy.random import seed
  from numpy.random import randn
  from scipy.stats import mannwhitneyu

# seed the random number generator
  seed(1)

# actives and inactives
  selection = [descriptor, 'bioactivity_class']
  df = df_2class[selection]
  active = df[df.bioactivity_class == 'active']
  active = active[descriptor]

  selection = [descriptor, 'bioactivity_class']
  df = df_2class[selection]
  inactive = df[df.bioactivity_class == 'inactive']
  inactive = inactive[descriptor]

# compare samples
  stat, p = mannwhitneyu(active, inactive)
  #print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
  alpha = 0.05
  if p > alpha:
    interpretation = 'Same distribution (fail to reject H0)'
  else:
    interpretation = 'Different distribution (reject H0)'
  
  results = pd.DataFrame({'Descriptor':descriptor,
                          'Statistics':stat,
                          'p':p,
                          'alpha':alpha,
                          'Interpretation':interpretation}, index=[0])
  filename = 'mannwhitneyu_' + descriptor + '.csv'
  results.to_csv(filename)

  return results


# In[35]:


#compare mannwhitney to the 'pIC50'
mannwhitney('pIC50')


# In[32]:


#MW

plt.figure(figsize=(5.5, 5.5))

sns.boxplot(x = 'bioactivity_class', y = 'MW', data = df_2class)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('MW', fontsize=14, fontweight='bold')

plt.savefig('/lakehouse/default/Files/plot_MW.pdf')


# In[36]:


#compare mannwhitney to the 'MW'
mannwhitney('MW')


# In[33]:


#LogP

plt.figure(figsize=(5.5, 5.5))

sns.boxplot(x = 'bioactivity_class', y = 'LogP', data = df_2class)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')

plt.savefig('/lakehouse/default/Files/plot_LogP.pdf')


# In[37]:


#compare mannwhitney to the 'MW'
mannwhitney('LogP')


# In[27]:


# NumHDonors

plt.figure(figsize=(5.5, 5.5))

sns.boxplot(x = 'bioactivity_class', y = 'NumHDonors', data = df_2class)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('NumHDonors', fontsize=14, fontweight='bold')

plt.savefig('plot_NumHDonors.pdf')


# In[38]:


#compare mannwhitney to the 'MW'
mannwhitney('NumHDonors')


# In[34]:


#NumHAcceptors

plt.figure(figsize=(5.5, 5.5))

sns.boxplot(x = 'bioactivity_class', y = 'NumHAcceptors', data = df_2class)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('NumHAcceptors', fontsize=14, fontweight='bold')

plt.savefig('/lakehouse/default/Files/plot_NumHAcceptors.pdf')


# In[39]:


#compare mannwhitney to the 'MW'
mannwhitney('NumHAcceptors')


# #### **Interpretation of Statistical Results**

# ##### **Box Plots**
# 
# ###### **pIC50 values**
# 
# Taking a look at pIC50 values, the **actives** and **inactives** displayed ***statistically significant difference***, which is to be expected since threshold values (``IC50 < 1,000 nM = Actives while IC50 > 10,000 nM = Inactives``, corresponding to ``pIC50 > 6 = Actives and pIC50 < 5 = Inactives``) were used to define actives and inactives.
# 
# ###### **Lipinski's descriptors**
# 
# Of the 4 Lipinski's descriptors (MW, LogP, NumHDonors and NumHAcceptors), only LogP exhibited ***no difference*** between the **actives** and **inactives** while the other 3 descriptors (MW, NumHDonors and NumHAcceptors) shows ***statistically significant difference*** between **actives** and **inactives**.

# In[44]:


import os
os.listdir()


# In[45]:


lakehouse_path = '/lakehouse/default/Files/results.zip'

with open('results.zip', 'rb') as f:
    data = f.read()

with open(lakehouse_path, 'wb') as f:
    f.write(data)


# In[43]:


#upload result in a zip file

get_ipython().system(' zip results.zip *.csv *.pdf')


# #### **Calculating Molecular Descriptors**
# Which are essential factors to be used in modeL building

# In[1]:


## Download PaDEL-Descriptor


get_ipython().system('git clone https://github.com/cdk/PaDEL-Descriptor.git')

get_ipython().system(' wget https://github.com/dataprofessor/bioinformatics/raw/master/padel.zip')
get_ipython().system(' wget https://github.com/dataprofessor/bioinformatics/raw/master/padel.sh')


# In[2]:


# Download the zip file
#!wget https://github.com/dataprofessor/bioinformatics/raw/master/padel.zip

## Download PaDEL-Descriptor

get_ipython().system('git clone https://github.com/cdk/PaDEL-Descriptor.git')


# In[ ]:


#Open Padel repo

get_ipython().system('wget https://github.com/dataprofessor/padel/raw/main/fingerprints_xml.zip')


# In[ ]:


# Confirm the file was downloaded
get_ipython().system('ls -lh padel.zip')


# In[17]:


#Open PaDEL-Descriptor repo

get_ipython().system('ls PaDEL-Descriptor')

get_ipython().system('wget https://github.com/cdk/PaDEL-Descriptor/archive/refs/heads/master.zip -O padel.zip')

get_ipython().system('unzip padel.zip')


# In[7]:


# Confirm the file was downloaded
get_ipython().system('ls -lh padel.zip')


# In[9]:


#checking the folders

get_ipython().system('ls PaDEL-Descriptor')
get_ipython().system('ls PaDEL-Descriptor-main')


# In[10]:


#search for the shell file

get_ipython().system('cat PaDEL-Descriptor/padel.sh')

#another method to search
get_ipython().system('find PaDEL-Descriptor -type f -name "*.sh"')


# ## **Load bioactivity data**
# 
# Download the curated ChEMBL bioactivity data that has been pre-processed. We will be using the **bioactivity_data_3class_pIC50.csv** file that essentially contain the pIC50 values that we will be using for building a regression model.

# In[28]:


get_ipython().system(' wget https://raw.githubusercontent.com/dataprofessor/data/master/acetylcholinesterase_04_bioactivity_data_3class_pIC50.csv')


# In[30]:


import pandas as pd

df3 = pd.read_csv('acetylcholinesterase_04_bioactivity_data_3class_pIC50.csv')

df3.head()


# In[31]:


selection = ['canonical_smiles','molecule_chembl_id']
df3_selection = df3[selection]
df3_selection.to_csv('molecule.smi', sep='\t', index=False, header=False)


# In[32]:


df3_selection.to_csv('/lakehouse/default/Files/molecule.smi', sep='\t', index=False, header=False)


# In[33]:


get_ipython().system(' cat molecule.smi | head -5')


# In[34]:


get_ipython().system(' cat molecule.smi | wc -l')


# In[35]:


# Calculate fingerprint descriptors
# Calculate PaDEL descriptors

get_ipython().system(' cat padel.sh')


# In[21]:


import pandas as pd
# Load data into pandas DataFrame from "/lakehouse/default/Files/descriptors_output.csv"
df3_X = pd.read_csv("/lakehouse/default/Files/descriptors_output.csv")

df3_X.head()


# #### **Preparing the X and Y Data Matrices**

# In[25]:


#X data variable

df3_X.head()


# In[26]:


#Drop the 'Name'column

df3_X = df3_X.drop(columns=['Name'])
df3_X.head()


# In[37]:


# Y variable

df3_Y = df3['pIC50']
df3_Y.head()


# In[38]:


# Combining X and Y variable

dataset3 = pd.concat([df3_X,df3_Y], axis=1)
dataset3


# In[39]:


#load dataset to Lakehouse

dataset3.to_csv('/lakehouse/default/Files/acetylcholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv', sep='\t', index=False, header=False)


# #### **Building a Simple regression model**
# 
# Using acetylcholinesterase inhibitors using the random forest algorithm

# In[43]:


#Import model libraries

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# In[47]:


#Connect to the githiub for the data

get_ipython().system(' wget https://github.com/dataprofessor/data/raw/master/acetylcholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv')


# In[50]:


#load the data

df = pd.read_csv('acetylcholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv')

df.head()


# In[51]:


#Input features
# The Acetylcholinesterase data set contains 882 input features and 1 output variable (pIC50 values)

X = df.drop('pIC50', axis=1)
X.head()


# In[53]:


#Outout feature

y = df.pIC50
y.head()


# In[56]:


#shape of y
y.shape


# In[57]:


#shape of X
X.shape


# In[58]:


#Remove low variance features

from sklearn.feature_selection import VarianceThreshold
selection = VarianceThreshold(threshold=(.8 * (1 - .8)))    
X = selection.fit_transform(X)


# In[59]:


#shape of X redefined
X.shape


# In[60]:


#Data split (80/20 ratio)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)


# In[61]:


#check the shape of trainin data

X_train.shape, Y_train.shape


# In[62]:


#check the shape of testing data 
X_test.shape, Y_test.shape


# In[63]:


#Building a Regression Model using Random Forest

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, Y_train)
r2 = model.score(X_test, Y_test)
r2


# In[64]:


#Building a Regression Model using Random Forest with seed number of 100

np.random.seed(100)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, Y_train)
r2 = model.score(X_test, Y_test)
r2


# In[67]:


#Scatter Plot of Experimental vs Predicted pIC50 Values

sns.set(color_codes=True)
sns.set_style("white")

Y_pred = model.predict(X_test)

# Use keyword arguments for x and y
ax = sns.regplot(x=Y_test, y=Y_pred, scatter_kws={'alpha': 0.4})
ax.set_xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
ax.set_ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.figure.set_size_inches(5, 5)
plt.show()


# ### **Model comparison**

# In[68]:


#Import libraries

get_ipython().system(' pip install lazypredict')


# In[69]:


#import important libraries

from sklearn.model_selection import train_test_split
import lazypredict
from lazypredict.Supervised import LazyRegressor


# In[70]:


#Loading dataset

get_ipython().system(' wget https://github.com/dataprofessor/data/raw/master/acetylcholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv')


# In[71]:


#assign variable

df = pd.read_csv('acetylcholinesterase_06_bioactivity_data_3class_pIC50_pubchem_fp.csv')

#Split data into X and Y
X = df.drop('pIC50', axis=1)
Y = df.pIC50


# In[72]:


#Examine X
X.shape


# In[73]:


# Remove low variance features
from sklearn.feature_selection import VarianceThreshold
selection = VarianceThreshold(threshold=(.8 * (1 - .8)))    
X = selection.fit_transform(X)
X.shape


# In[74]:


# Perform data splitting using 80/20 ratio
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[76]:


# Defines and builds the lazyclassifier (39 models)
clf = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
models_train,predictions_train = clf.fit(X_train, X_train, Y_train, Y_train)
models_test,predictions_test = clf.fit(X_train, X_test, Y_train, Y_test)


# In[77]:


# Performance table of the training set (80% subset)
predictions_train


# In[78]:


# Performance table of the test set (20% subset)
predictions_test


# #### **Data visualization of model performance**

# In[79]:


# Bar plot of R-squared values
import matplotlib.pyplot as plt
import seaborn as sns

#train["R-Squared"] = [0 if i < 0 else i for i in train.iloc[:,0] ]

plt.figure(figsize=(5, 10))
sns.set_theme(style="whitegrid")
ax = sns.barplot(y=predictions_train.index, x="R-Squared", data=predictions_train)
ax.set(xlim=(0, 1))
     


# In[80]:


# Bar plot of RMSE values
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5, 10))
sns.set_theme(style="whitegrid")
ax = sns.barplot(y=predictions_train.index, x="RMSE", data=predictions_train)
ax.set(xlim=(0, 10))


# In[81]:


# Bar plot of calculation time
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(5, 10))
sns.set_theme(style="whitegrid")
ax = sns.barplot(y=predictions_train.index, x="Time Taken", data=predictions_train)
ax.set(xlim=(0, 10))

