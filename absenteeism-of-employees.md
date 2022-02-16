```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

    /kaggle/input/absenteeism_data.csv
    


```python
raw_csv_data = pd.read_csv("/kaggle/input/absenteeism_data.csv")

raw_csv_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Reason for Absence</th>
      <th>Date</th>
      <th>Transportation Expense</th>
      <th>Distance to Work</th>
      <th>Age</th>
      <th>Daily Work Load Average</th>
      <th>Body Mass Index</th>
      <th>Education</th>
      <th>Children</th>
      <th>Pets</th>
      <th>Absenteeism Time in Hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11</td>
      <td>26</td>
      <td>07-07-2015</td>
      <td>289</td>
      <td>36</td>
      <td>33</td>
      <td>239.554</td>
      <td>30</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>36</td>
      <td>0</td>
      <td>14-07-2015</td>
      <td>118</td>
      <td>13</td>
      <td>50</td>
      <td>239.554</td>
      <td>31</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>23</td>
      <td>15-07-2015</td>
      <td>179</td>
      <td>51</td>
      <td>38</td>
      <td>239.554</td>
      <td>31</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>7</td>
      <td>16-07-2015</td>
      <td>279</td>
      <td>5</td>
      <td>39</td>
      <td>239.554</td>
      <td>24</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>23</td>
      <td>23-07-2015</td>
      <td>289</td>
      <td>36</td>
      <td>33</td>
      <td>239.554</td>
      <td>30</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



## Preprocessing the data


```python
df = raw_csv_data.copy()

# Setting maximum number of rows displayed in output
pd.options.display.max_rows = 3
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 699 entries, 0 to 698
    Data columns (total 12 columns):
     #   Column                     Non-Null Count  Dtype  
    ---  ------                     --------------  -----  
     0   ID                         699 non-null    int64  
     1   Reason for Absence         699 non-null    int64  
     2   Date                       699 non-null    object 
     3   Transportation Expense     699 non-null    int64  
     4   Distance to Work           699 non-null    int64  
     5   Age                        699 non-null    int64  
     6   Daily Work Load Average    699 non-null    float64
     7   Body Mass Index            699 non-null    int64  
     8   Education                  699 non-null    int64  
     9   Children                   699 non-null    int64  
     10  Pets                       699 non-null    int64  
     11  Absenteeism Time in Hours  699 non-null    int64  
    dtypes: float64(1), int64(10), object(1)
    memory usage: 65.7+ KB
    

Dependent variables in dataset - reason for absence, transportation expense, distance to work, age, daily work load average, BMI, education, children, pets
Independent variable - absenteeism time in hrs

#### Dropping the irrelevant variables (ID) to our analysis


```python
df = df.drop(['ID'], axis = 1)

display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reason for Absence</th>
      <th>Date</th>
      <th>Transportation Expense</th>
      <th>Distance to Work</th>
      <th>Age</th>
      <th>Daily Work Load Average</th>
      <th>Body Mass Index</th>
      <th>Education</th>
      <th>Children</th>
      <th>Pets</th>
      <th>Absenteeism Time in Hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>26</td>
      <td>07-07-2015</td>
      <td>289</td>
      <td>36</td>
      <td>33</td>
      <td>239.554</td>
      <td>30</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>698</th>
      <td>28</td>
      <td>31-05-2018</td>
      <td>291</td>
      <td>31</td>
      <td>40</td>
      <td>237.656</td>
      <td>25</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>699 rows × 11 columns</p>
</div>


#### Analyzing the reason for absence


```python
display(df['Reason for Absence'].min())
display(df['Reason for Absence'].max())

# Checking all unique values of 'Reason of Absence'
display(pd.unique(df['Reason for Absence']))

# Counting the length of array returned
display(len(pd.unique(df['Reason for Absence'])))
```


    0



    28



    array([26,  0, 23,  7, 22, 19,  1, 11, 14, 21, 10, 13, 28, 18, 25, 24,  6,
           27, 17,  8, 12,  5,  9, 15,  4,  3,  2, 16])



    28



```python
# Sorting array to find out the missing value
sorted(df['Reason for Absence'].unique())
```




    [0,
     1,
     2,
     3,
     4,
     5,
     6,
     7,
     8,
     9,
     10,
     11,
     12,
     13,
     14,
     15,
     16,
     17,
     18,
     19,
     21,
     22,
     23,
     24,
     25,
     26,
     27,
     28]



#### Create dummy variables for 'Reason for Absence'


```python
reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)
```


```python
# Checking if an employee was absent from work for more than one reason
reason_columns['check'] = reason_columns.sum(axis = 1)

reason_columns
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>19</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>check</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>698</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>699 rows × 28 columns</p>
</div>




```python
# Calculating sum of check column
display(reason_columns['check'].sum(axis = 0))

# Displaying unique values in check column
display(reason_columns['check'].unique())
```


    661



    array([1, 0])



```python
# Dropping the check column
reason_columns = reason_columns.drop(['check'], axis = 1)

# Dropping 'Reason for Absence' to avoid multicollinearity
df = df.drop(['Reason for Absence'], axis = 1)
```

#### Classifying the 28 reasons into 4 groups


```python
reason_type_1 = reason_columns.iloc[:, 1:14].max(axis = 1)
reason_type_2 = reason_columns.iloc[:, 15:17].max(axis = 1)
reason_type_3 = reason_columns.iloc[:, 18:21].max(axis = 1)
reason_type_4 = reason_columns.iloc[:, 22:].max(axis = 1)
```

#### Concatenating reason types and df


```python
df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis = 1)

display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Transportation Expense</th>
      <th>Distance to Work</th>
      <th>Age</th>
      <th>Daily Work Load Average</th>
      <th>Body Mass Index</th>
      <th>Education</th>
      <th>Children</th>
      <th>Pets</th>
      <th>Absenteeism Time in Hours</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>07-07-2015</td>
      <td>289</td>
      <td>36</td>
      <td>33</td>
      <td>239.554</td>
      <td>30</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>698</th>
      <td>31-05-2018</td>
      <td>291</td>
      <td>31</td>
      <td>40</td>
      <td>237.656</td>
      <td>25</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>699 rows × 14 columns</p>
</div>



```python
# Renaming the columns 0, 1, 2, 4
column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4'] 
df.columns = column_names

# Reordering the columns
column_names_reordered = [ 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours'] 
df = df[column_names_reordered]
```

#### Creating a checkpoint


```python
df_reasons_mod = df.copy()

```

#### Analyzing the 'Date' column


```python
type(df_reasons_mod['Date'][0])
```




    str




```python
# Converting the column values to timestamp
df_reasons_mod['Date'] = pd.to_datetime(df_reasons_mod['Date'], format = '%d-%m-%Y')

df_reasons_mod['Date']
```




    0     2015-07-07
             ...    
    698   2018-05-31
    Name: Date, Length: 699, dtype: datetime64[ns]




```python
# Extracting the month values from 'Date' column
list_months = []
df_reasons_rows_length = df_reasons_mod.shape[0]
for i in range(df_reasons_rows_length):
    list_months.append(df_reasons_mod['Date'][i].month)
    
# Printing the length of the list_months
display(len(list_months))    
```


    699



```python
# Adding the list_months to df_reasons_mod
df_reasons_mod['Month Value'] = list_months

# Extracting the day of the week from 'Date'
def date_to_weekday(date_value):
    return date_value.weekday()

# Adding 'Day of the Week' column to the dataset by calling the date_to_weekday function
df_reasons_mod['Day of the Week'] = df_reasons_mod['Date'].apply(date_to_weekday)
```


```python
# Removing the 'Date' column
df_reasons_mod.drop(['Date'], axis = 1)

# Reordering the columns
column_names_reordered = [ 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value', 'Day of the Week', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours'] 
df_reasons_mod = df_reasons_mod[column_names_reordered]

# Printing df_reasons_mod
display(df_reasons_mod)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reason_1</th>
      <th>Reason_2</th>
      <th>Reason_3</th>
      <th>Reason_4</th>
      <th>Month Value</th>
      <th>Day of the Week</th>
      <th>Transportation Expense</th>
      <th>Distance to Work</th>
      <th>Age</th>
      <th>Daily Work Load Average</th>
      <th>Body Mass Index</th>
      <th>Education</th>
      <th>Children</th>
      <th>Pets</th>
      <th>Absenteeism Time in Hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>289</td>
      <td>36</td>
      <td>33</td>
      <td>239.554</td>
      <td>30</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>698</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>291</td>
      <td>31</td>
      <td>40</td>
      <td>237.656</td>
      <td>25</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>699 rows × 15 columns</p>
</div>


#### Creating a checkpoint


```python
df_date_reasons_mod = df_reasons_mod.copy()
```

#### Analyzing 'Education' column


```python
# Calculating the number of unique values in the 'Education' column
df_date_reasons_mod['Education'].unique()
```




    array([1, 3, 2, 4])




```python
# Setting maximum number of rows displayed in output
pd.options.display.max_rows = 4

# Counting the number of each value present in the column
df_date_reasons_mod['Education'].value_counts()
```




    1    582
    3     73
    2     40
    4      4
    Name: Education, dtype: int64



As the number of 2s, 3s, and 4s are less compared to 1s, we will group the last three values into one


```python
# Grouping values 2, 3, and 4 to one group
# Mapping 1 to 0 & 2, 3, 4 to 1
df_date_reasons_mod['Education'] = df_date_reasons_mod['Education'].map({1:0, 2:1, 3:1, 4:1})
```

#### Final Checkpoint


```python
df_preprocessed = df_date_reasons_mod.copy()

df_preprocessed
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reason_1</th>
      <th>Reason_2</th>
      <th>Reason_3</th>
      <th>Reason_4</th>
      <th>Month Value</th>
      <th>Day of the Week</th>
      <th>Transportation Expense</th>
      <th>Distance to Work</th>
      <th>Age</th>
      <th>Daily Work Load Average</th>
      <th>Body Mass Index</th>
      <th>Education</th>
      <th>Children</th>
      <th>Pets</th>
      <th>Absenteeism Time in Hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>289</td>
      <td>36</td>
      <td>33</td>
      <td>239.554</td>
      <td>30</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>118</td>
      <td>13</td>
      <td>50</td>
      <td>239.554</td>
      <td>31</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>697</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>235</td>
      <td>16</td>
      <td>32</td>
      <td>237.656</td>
      <td>25</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>698</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>291</td>
      <td>31</td>
      <td>40</td>
      <td>237.656</td>
      <td>25</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>699 rows × 15 columns</p>
</div>



Creating the targets


```python
# Calculating the mean of the 'Absenteeism Time in Hours' column values
df_preprocessed_median = df_preprocessed['Absenteeism Time in Hours'].median()
```


```python
# Classifying data into two groups, for 'Absenteeism Time in Hours' > median - class 1 otherwise 0
targets = np.where(df_preprocessed['Absenteeism Time in Hours'] > df_preprocessed_median, 1, 0)

# Adding targets to the dataframe 
df_preprocessed['Excessive Absenteeism'] = targets
```


```python
# Check if the groups have been divded almost equally, proceed if value is between 0.40 to 0.50 for accurate model creation
targets.sum() / targets.shape[0]
```




    0.4563662374821173




```python
# Dropping 'Absenteeism Time in Hours' column, and assigning the returned dataframe
data_with_targets = df_preprocessed.drop(['Absenteeism Time in Hours', 'Distance to Work', 'Daily Work Load Average', 'Age'], axis = 1)
```

### Select inputs for regression


```python
# Selecting first 14 columns as inputs
unscaled_inputs = data_with_targets.iloc[:, :-1]

# Printing the data
display(unscaled_inputs)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reason_1</th>
      <th>Reason_2</th>
      <th>Reason_3</th>
      <th>Reason_4</th>
      <th>Month Value</th>
      <th>Day of the Week</th>
      <th>Transportation Expense</th>
      <th>Body Mass Index</th>
      <th>Education</th>
      <th>Children</th>
      <th>Pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>289</td>
      <td>30</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>118</td>
      <td>31</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>697</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>235</td>
      <td>25</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>698</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>291</td>
      <td>25</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>699 rows × 11 columns</p>
</div>


Standardize the data


```python
# Importing StandardScaler, BaseEstimator, and TransformerMixin from sklean
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class CustomScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns, copy = True, with_mean = True, with_std = True):
        self.scaler = StandardScaler(copy, with_mean, with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None
    
    def fit(self, X, y = None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    def transform(self, X, y = None, copy = None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns = self.columns)
        X_not_scaled = X.iloc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis = 1)[init_col_order]
    
columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit] 
        
# Creating an empty scaler object
absenteeism_scaler = CustomScaler(columns_to_scale)
```


```python
# Calculating mean and standard deviation of each unscaled input
absenteeism_scaler.fit(unscaled_inputs)
```

    /opt/conda/lib/python3.7/site-packages/sklearn/base.py:197: FutureWarning: From version 0.24, get_params will raise an AttributeError if a parameter cannot be retrieved as an instance attribute. Previously it would return None.
      FutureWarning)
    




    CustomScaler(columns=['Month Value', 'Day of the Week',
                          'Transportation Expense', 'Body Mass Index', 'Education',
                          'Children', 'Pets'],
                 copy=None, with_mean=None, with_std=None)




```python
# Creating scaled inputs using absenteeism_scaler object
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)

# Printing scaled inputs
display(scaled_inputs)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Reason_1</th>
      <th>Reason_2</th>
      <th>Reason_3</th>
      <th>Reason_4</th>
      <th>Month Value</th>
      <th>Day of the Week</th>
      <th>Transportation Expense</th>
      <th>Body Mass Index</th>
      <th>Education</th>
      <th>Children</th>
      <th>Pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.183294</td>
      <td>-0.682176</td>
      <td>1.004498</td>
      <td>0.768869</td>
      <td>-0.448365</td>
      <td>0.879058</td>
      <td>0.267518</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.183294</td>
      <td>-0.682176</td>
      <td>-1.574973</td>
      <td>1.004073</td>
      <td>-0.448365</td>
      <td>-0.020593</td>
      <td>-0.590258</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>697</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.387407</td>
      <td>0.670564</td>
      <td>0.189928</td>
      <td>-0.407147</td>
      <td>2.230327</td>
      <td>-0.920243</td>
      <td>-0.590258</td>
    </tr>
    <tr>
      <th>698</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-0.387407</td>
      <td>0.670564</td>
      <td>1.034667</td>
      <td>-0.407147</td>
      <td>-0.448365</td>
      <td>-0.020593</td>
      <td>0.267518</td>
    </tr>
  </tbody>
</table>
<p>699 rows × 11 columns</p>
</div>


Shuffle and divide the data into train & test


```python
# Importing train_test_split
from sklearn.model_selection import train_test_split
```


```python
# Shuffling and dividing the data
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size = 0.8, shuffle = True, random_state = 5)

# Checking shapes train and test data
print(x_train.shape, y_train.shape, "&", x_test.shape, y_test.shape)
```

    (559, 11) (559,) & (140, 11) (140,)
    

## **Logistic Regression using Machine Learning library sklearn**


```python
# Importing LogisticRegression and matrics from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
```

Training the model


```python
# Creating object of LogisticRegression
regression = LogisticRegression()

# Creating the model using machine learning
regression.fit(x_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)




```python
# Calculating accuracy of the model
print("Accuracy: ", regression.score(x_train, y_train))

# Calculationg accuracy manually
model_output = regression.predict(x_train)
correct_predictions = np.sum(model_output == y_train) # Calculating sum of correctly predicted outputs
manual_accuracy = correct_predictions / model_output.shape[0]
print("Accuracy by manual calculation: ", manual_accuracy)
```

    Accuracy:  0.7513416815742398
    Accuracy by manual calculation:  0.7513416815742398
    


```python
# Finding intercepts and coefficiants
print("Intercepts", regression.intercept_)
print("Coefficiants", regression.coef_)

# Creating table to map coefficiants with indepdent variable values
feature_names = unscaled_inputs.columns.values 

# Creating summary
summary = pd.DataFrame(columns = ['Feature Names'], data = feature_names)
summary['Coefficients'] = np.transpose(regression.coef_)

# Adding intercept to the summary
summary.index = summary.index + 1
summary.loc[0] = ['Intercept', regression.intercept_[0]]

# Sorting my index
summary = summary.sort_index()
display(summary)
```

    Intercepts [-0.80301785]
    Coefficiants [[ 1.82954524 -0.29577508  2.21488453 -0.18015217  0.09219541 -0.27972192
       0.45541654  0.09492564  0.11584819  0.43397297 -0.36155211]]
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature Names</th>
      <th>Coefficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Intercept</td>
      <td>-0.803018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Reason_1</td>
      <td>1.829545</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Children</td>
      <td>0.433973</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Pets</td>
      <td>-0.361552</td>
    </tr>
  </tbody>
</table>
<p>12 rows × 2 columns</p>
</div>


#### Studying the model coefficiants


```python
# Adding odds ratio column
summary['Odds Ratio'] = np.exp(summary['Coefficients'])

# Setting maximum number of rows displayed in output
pd.options.display.max_rows = None

# Sorting summary table
summary.sort_values('Odds Ratio', ascending = False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature Names</th>
      <th>Coefficients</th>
      <th>Odds Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Reason_3</td>
      <td>2.214885</td>
      <td>9.160351</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Reason_1</td>
      <td>1.829545</td>
      <td>6.231052</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Transportation Expense</td>
      <td>0.455417</td>
      <td>1.576830</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Children</td>
      <td>0.433973</td>
      <td>1.543377</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Education</td>
      <td>0.115848</td>
      <td>1.122825</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Body Mass Index</td>
      <td>0.094926</td>
      <td>1.099577</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Month Value</td>
      <td>0.092195</td>
      <td>1.096579</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Reason_4</td>
      <td>-0.180152</td>
      <td>0.835143</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Day of the Week</td>
      <td>-0.279722</td>
      <td>0.755994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Reason_2</td>
      <td>-0.295775</td>
      <td>0.743955</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Pets</td>
      <td>-0.361552</td>
      <td>0.696594</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Intercept</td>
      <td>-0.803018</td>
      <td>0.447975</td>
    </tr>
  </tbody>
</table>
</div>



## Testing the model


```python
# Testing the accuracy with the testing data
display(regression.score(x_test, y_test))
```


    0.6857142857142857



```python
# Finding estimates for possible outputs with x_test as input
predicted_proba = regression.predict_proba(x_test)

display(predicted_proba[:,1])
```


    array([0.78866967, 0.40377287, 0.93723839, 0.71712195, 0.69153393,
           0.64377733, 0.22855656, 0.17590453, 0.17532485, 0.17211057,
           0.15912522, 0.8938425 , 0.55840089, 0.6460777 , 0.28817229,
           0.1466337 , 0.24165084, 0.87694228, 0.07667595, 0.3580118 ,
           0.40386486, 0.16869974, 0.23928916, 0.64656505, 0.15829953,
           0.84897568, 0.54537747, 0.69045794, 0.31003333, 0.81724814,
           0.24374028, 0.76242583, 0.2202764 , 0.5115144 , 0.1544825 ,
           0.15416738, 0.15608692, 0.2411538 , 0.40890341, 0.90920237,
           0.56486606, 0.31237339, 0.80721754, 0.598641  , 0.8312926 ,
           0.82776821, 0.32628336, 0.36524827, 0.70628642, 0.8087439 ,
           0.63397683, 0.67088781, 0.11412049, 0.86735033, 0.64978742,
           0.76318701, 0.29340836, 0.25858088, 0.80721754, 0.8529218 ,
           0.54972459, 0.79985717, 0.09364933, 0.84897706, 0.1466337 ,
           0.59068584, 0.47598312, 0.18701644, 0.4742865 , 0.55611114,
           0.59931252, 0.81568061, 0.2878127 , 0.19089448, 0.20480122,
           0.16873265, 0.92904123, 0.31125013, 0.34320212, 0.8968614 ,
           0.17826131, 0.50127165, 0.30810218, 0.68589395, 0.71605827,
           0.50353242, 0.23322824, 0.79113327, 0.76172446, 0.22477478,
           0.58318326, 0.20114935, 0.7967774 , 0.21749595, 0.27395102,
           0.64856998, 0.3612062 , 0.47239973, 0.71712136, 0.30563857,
           0.72238838, 0.60554006, 0.26361096, 0.22521943, 0.20393692,
           0.82776821, 0.59985119, 0.24556984, 0.85267127, 0.78915681,
           0.22202206, 0.50840786, 0.50604035, 0.37221328, 0.21053365,
           0.27453656, 0.33702557, 0.5323036 , 0.29889202, 0.73281577,
           0.14090143, 0.22311826, 0.62223014, 0.30048461, 0.9618354 ,
           0.09634417, 0.58969909, 0.17212332, 0.47738701, 0.28997419,
           0.50382789, 0.1435336 , 0.69598748, 0.12546346, 0.12837859,
           0.1423031 , 0.51639495, 0.17242141, 0.47611079, 0.3131416 ])

