
# House Price Prediction through Regression

### Through the Ames dataset



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LogisticRegression as logR
from sklearn.linear_model import ElasticNet

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb

import seaborn as sns
%matplotlib inline


#https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data


```

### First let's load both datasets and take some quick looks at the data


```python
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


```


```python
print(df_train.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
    Id               1460 non-null int64
    MSSubClass       1460 non-null int64
    MSZoning         1460 non-null object
    LotFrontage      1201 non-null float64
    LotArea          1460 non-null int64
    Street           1460 non-null object
    Alley            91 non-null object
    LotShape         1460 non-null object
    LandContour      1460 non-null object
    Utilities        1460 non-null object
    LotConfig        1460 non-null object
    LandSlope        1460 non-null object
    Neighborhood     1460 non-null object
    Condition1       1460 non-null object
    Condition2       1460 non-null object
    BldgType         1460 non-null object
    HouseStyle       1460 non-null object
    OverallQual      1460 non-null int64
    OverallCond      1460 non-null int64
    YearBuilt        1460 non-null int64
    YearRemodAdd     1460 non-null int64
    RoofStyle        1460 non-null object
    RoofMatl         1460 non-null object
    Exterior1st      1460 non-null object
    Exterior2nd      1460 non-null object
    MasVnrType       1452 non-null object
    MasVnrArea       1452 non-null float64
    ExterQual        1460 non-null object
    ExterCond        1460 non-null object
    Foundation       1460 non-null object
    BsmtQual         1423 non-null object
    BsmtCond         1423 non-null object
    BsmtExposure     1422 non-null object
    BsmtFinType1     1423 non-null object
    BsmtFinSF1       1460 non-null int64
    BsmtFinType2     1422 non-null object
    BsmtFinSF2       1460 non-null int64
    BsmtUnfSF        1460 non-null int64
    TotalBsmtSF      1460 non-null int64
    Heating          1460 non-null object
    HeatingQC        1460 non-null object
    CentralAir       1460 non-null object
    Electrical       1459 non-null object
    1stFlrSF         1460 non-null int64
    2ndFlrSF         1460 non-null int64
    LowQualFinSF     1460 non-null int64
    GrLivArea        1460 non-null int64
    BsmtFullBath     1460 non-null int64
    BsmtHalfBath     1460 non-null int64
    FullBath         1460 non-null int64
    HalfBath         1460 non-null int64
    BedroomAbvGr     1460 non-null int64
    KitchenAbvGr     1460 non-null int64
    KitchenQual      1460 non-null object
    TotRmsAbvGrd     1460 non-null int64
    Functional       1460 non-null object
    Fireplaces       1460 non-null int64
    FireplaceQu      770 non-null object
    GarageType       1379 non-null object
    GarageYrBlt      1379 non-null float64
    GarageFinish     1379 non-null object
    GarageCars       1460 non-null int64
    GarageArea       1460 non-null int64
    GarageQual       1379 non-null object
    GarageCond       1379 non-null object
    PavedDrive       1460 non-null object
    WoodDeckSF       1460 non-null int64
    OpenPorchSF      1460 non-null int64
    EnclosedPorch    1460 non-null int64
    3SsnPorch        1460 non-null int64
    ScreenPorch      1460 non-null int64
    PoolArea         1460 non-null int64
    PoolQC           7 non-null object
    Fence            281 non-null object
    MiscFeature      54 non-null object
    MiscVal          1460 non-null int64
    MoSold           1460 non-null int64
    YrSold           1460 non-null int64
    SaleType         1460 non-null object
    SaleCondition    1460 non-null object
    SalePrice        1460 non-null int64
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB
    None



```python
print(df_test.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1459 entries, 0 to 1458
    Data columns (total 80 columns):
    Id               1459 non-null int64
    MSSubClass       1459 non-null int64
    MSZoning         1455 non-null object
    LotFrontage      1232 non-null float64
    LotArea          1459 non-null int64
    Street           1459 non-null object
    Alley            107 non-null object
    LotShape         1459 non-null object
    LandContour      1459 non-null object
    Utilities        1457 non-null object
    LotConfig        1459 non-null object
    LandSlope        1459 non-null object
    Neighborhood     1459 non-null object
    Condition1       1459 non-null object
    Condition2       1459 non-null object
    BldgType         1459 non-null object
    HouseStyle       1459 non-null object
    OverallQual      1459 non-null int64
    OverallCond      1459 non-null int64
    YearBuilt        1459 non-null int64
    YearRemodAdd     1459 non-null int64
    RoofStyle        1459 non-null object
    RoofMatl         1459 non-null object
    Exterior1st      1458 non-null object
    Exterior2nd      1458 non-null object
    MasVnrType       1443 non-null object
    MasVnrArea       1444 non-null float64
    ExterQual        1459 non-null object
    ExterCond        1459 non-null object
    Foundation       1459 non-null object
    BsmtQual         1415 non-null object
    BsmtCond         1414 non-null object
    BsmtExposure     1415 non-null object
    BsmtFinType1     1417 non-null object
    BsmtFinSF1       1458 non-null float64
    BsmtFinType2     1417 non-null object
    BsmtFinSF2       1458 non-null float64
    BsmtUnfSF        1458 non-null float64
    TotalBsmtSF      1458 non-null float64
    Heating          1459 non-null object
    HeatingQC        1459 non-null object
    CentralAir       1459 non-null object
    Electrical       1459 non-null object
    1stFlrSF         1459 non-null int64
    2ndFlrSF         1459 non-null int64
    LowQualFinSF     1459 non-null int64
    GrLivArea        1459 non-null int64
    BsmtFullBath     1457 non-null float64
    BsmtHalfBath     1457 non-null float64
    FullBath         1459 non-null int64
    HalfBath         1459 non-null int64
    BedroomAbvGr     1459 non-null int64
    KitchenAbvGr     1459 non-null int64
    KitchenQual      1458 non-null object
    TotRmsAbvGrd     1459 non-null int64
    Functional       1457 non-null object
    Fireplaces       1459 non-null int64
    FireplaceQu      729 non-null object
    GarageType       1383 non-null object
    GarageYrBlt      1381 non-null float64
    GarageFinish     1381 non-null object
    GarageCars       1458 non-null float64
    GarageArea       1458 non-null float64
    GarageQual       1381 non-null object
    GarageCond       1381 non-null object
    PavedDrive       1459 non-null object
    WoodDeckSF       1459 non-null int64
    OpenPorchSF      1459 non-null int64
    EnclosedPorch    1459 non-null int64
    3SsnPorch        1459 non-null int64
    ScreenPorch      1459 non-null int64
    PoolArea         1459 non-null int64
    PoolQC           3 non-null object
    Fence            290 non-null object
    MiscFeature      51 non-null object
    MiscVal          1459 non-null int64
    MoSold           1459 non-null int64
    YrSold           1459 non-null int64
    SaleType         1458 non-null object
    SaleCondition    1459 non-null object
    dtypes: float64(11), int64(26), object(43)
    memory usage: 912.0+ KB
    None


### Let's take a look at the sale price, the variable we are to model and predict


```python
plt.hist(df_train['SalePrice'], 100)

```




    (array([  5.,   0.,   5.,   6.,   6.,   7.,  32.,  29.,  23.,  35.,  59.,
             67.,  73.,  92.,  89.,  91.,  62.,  60.,  57.,  73.,  67.,  54.,
             45.,  33.,  30.,  31.,  31.,  33.,  26.,  23.,  17.,  19.,  18.,
             18.,  14.,  11.,   5.,   8.,  13.,  12.,  10.,   6.,   6.,   5.,
              1.,   3.,   5.,   6.,   5.,   4.,   3.,   2.,   3.,   1.,   3.,
              1.,   2.,   2.,   0.,   2.,   0.,   1.,   1.,   0.,   1.,   0.,
              0.,   0.,   0.,   1.,   0.,   0.,   2.,   0.,   0.,   0.,   1.,
              0.,   0.,   0.,   1.,   1.,   0.,   0.,   0.,   0.,   0.,   0.,
              0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   1.,
              1.]),
     array([  34900.,   42101.,   49302.,   56503.,   63704.,   70905.,
              78106.,   85307.,   92508.,   99709.,  106910.,  114111.,
             121312.,  128513.,  135714.,  142915.,  150116.,  157317.,
             164518.,  171719.,  178920.,  186121.,  193322.,  200523.,
             207724.,  214925.,  222126.,  229327.,  236528.,  243729.,
             250930.,  258131.,  265332.,  272533.,  279734.,  286935.,
             294136.,  301337.,  308538.,  315739.,  322940.,  330141.,
             337342.,  344543.,  351744.,  358945.,  366146.,  373347.,
             380548.,  387749.,  394950.,  402151.,  409352.,  416553.,
             423754.,  430955.,  438156.,  445357.,  452558.,  459759.,
             466960.,  474161.,  481362.,  488563.,  495764.,  502965.,
             510166.,  517367.,  524568.,  531769.,  538970.,  546171.,
             553372.,  560573.,  567774.,  574975.,  582176.,  589377.,
             596578.,  603779.,  610980.,  618181.,  625382.,  632583.,
             639784.,  646985.,  654186.,  661387.,  668588.,  675789.,
             682990.,  690191.,  697392.,  704593.,  711794.,  718995.,
             726196.,  733397.,  740598.,  747799.,  755000.]),
     <a list of 100 Patch objects>)




![png](House%20Price%20Prediction_files/House%20Price%20Prediction_7_1.png)


### First we clean the dataset.  Let's combine the two first so we can perform the same operations on both.  We will split it up later.


```python
df = pd.concat([df_train, df_test])

print(df.info())
df.head()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2919 entries, 0 to 1458
    Data columns (total 81 columns):
    1stFlrSF         2919 non-null int64
    2ndFlrSF         2919 non-null int64
    3SsnPorch        2919 non-null int64
    Alley            198 non-null object
    BedroomAbvGr     2919 non-null int64
    BldgType         2919 non-null object
    BsmtCond         2837 non-null object
    BsmtExposure     2837 non-null object
    BsmtFinSF1       2918 non-null float64
    BsmtFinSF2       2918 non-null float64
    BsmtFinType1     2840 non-null object
    BsmtFinType2     2839 non-null object
    BsmtFullBath     2917 non-null float64
    BsmtHalfBath     2917 non-null float64
    BsmtQual         2838 non-null object
    BsmtUnfSF        2918 non-null float64
    CentralAir       2919 non-null object
    Condition1       2919 non-null object
    Condition2       2919 non-null object
    Electrical       2918 non-null object
    EnclosedPorch    2919 non-null int64
    ExterCond        2919 non-null object
    ExterQual        2919 non-null object
    Exterior1st      2918 non-null object
    Exterior2nd      2918 non-null object
    Fence            571 non-null object
    FireplaceQu      1499 non-null object
    Fireplaces       2919 non-null int64
    Foundation       2919 non-null object
    FullBath         2919 non-null int64
    Functional       2917 non-null object
    GarageArea       2918 non-null float64
    GarageCars       2918 non-null float64
    GarageCond       2760 non-null object
    GarageFinish     2760 non-null object
    GarageQual       2760 non-null object
    GarageType       2762 non-null object
    GarageYrBlt      2760 non-null float64
    GrLivArea        2919 non-null int64
    HalfBath         2919 non-null int64
    Heating          2919 non-null object
    HeatingQC        2919 non-null object
    HouseStyle       2919 non-null object
    Id               2919 non-null int64
    KitchenAbvGr     2919 non-null int64
    KitchenQual      2918 non-null object
    LandContour      2919 non-null object
    LandSlope        2919 non-null object
    LotArea          2919 non-null int64
    LotConfig        2919 non-null object
    LotFrontage      2433 non-null float64
    LotShape         2919 non-null object
    LowQualFinSF     2919 non-null int64
    MSSubClass       2919 non-null int64
    MSZoning         2915 non-null object
    MasVnrArea       2896 non-null float64
    MasVnrType       2895 non-null object
    MiscFeature      105 non-null object
    MiscVal          2919 non-null int64
    MoSold           2919 non-null int64
    Neighborhood     2919 non-null object
    OpenPorchSF      2919 non-null int64
    OverallCond      2919 non-null int64
    OverallQual      2919 non-null int64
    PavedDrive       2919 non-null object
    PoolArea         2919 non-null int64
    PoolQC           10 non-null object
    RoofMatl         2919 non-null object
    RoofStyle        2919 non-null object
    SaleCondition    2919 non-null object
    SalePrice        1460 non-null float64
    SaleType         2918 non-null object
    ScreenPorch      2919 non-null int64
    Street           2919 non-null object
    TotRmsAbvGrd     2919 non-null int64
    TotalBsmtSF      2918 non-null float64
    Utilities        2917 non-null object
    WoodDeckSF       2919 non-null int64
    YearBuilt        2919 non-null int64
    YearRemodAdd     2919 non-null int64
    YrSold           2919 non-null int64
    dtypes: float64(12), int64(26), object(43)
    memory usage: 1.8+ MB
    None





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>3SsnPorch</th>
      <th>Alley</th>
      <th>BedroomAbvGr</th>
      <th>BldgType</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>...</th>
      <th>SaleType</th>
      <th>ScreenPorch</th>
      <th>Street</th>
      <th>TotRmsAbvGrd</th>
      <th>TotalBsmtSF</th>
      <th>Utilities</th>
      <th>WoodDeckSF</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>YrSold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>856</td>
      <td>854</td>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
      <td>1Fam</td>
      <td>TA</td>
      <td>No</td>
      <td>706.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>WD</td>
      <td>0</td>
      <td>Pave</td>
      <td>8</td>
      <td>856.0</td>
      <td>AllPub</td>
      <td>0</td>
      <td>2003</td>
      <td>2003</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1262</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
      <td>1Fam</td>
      <td>TA</td>
      <td>Gd</td>
      <td>978.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>WD</td>
      <td>0</td>
      <td>Pave</td>
      <td>6</td>
      <td>1262.0</td>
      <td>AllPub</td>
      <td>298</td>
      <td>1976</td>
      <td>1976</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>2</th>
      <td>920</td>
      <td>866</td>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
      <td>1Fam</td>
      <td>TA</td>
      <td>Mn</td>
      <td>486.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>WD</td>
      <td>0</td>
      <td>Pave</td>
      <td>6</td>
      <td>920.0</td>
      <td>AllPub</td>
      <td>0</td>
      <td>2001</td>
      <td>2002</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>3</th>
      <td>961</td>
      <td>756</td>
      <td>0</td>
      <td>NaN</td>
      <td>3</td>
      <td>1Fam</td>
      <td>Gd</td>
      <td>No</td>
      <td>216.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>WD</td>
      <td>0</td>
      <td>Pave</td>
      <td>7</td>
      <td>756.0</td>
      <td>AllPub</td>
      <td>0</td>
      <td>1915</td>
      <td>1970</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1145</td>
      <td>1053</td>
      <td>0</td>
      <td>NaN</td>
      <td>4</td>
      <td>1Fam</td>
      <td>TA</td>
      <td>Av</td>
      <td>655.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>WD</td>
      <td>0</td>
      <td>Pave</td>
      <td>9</td>
      <td>1145.0</td>
      <td>AllPub</td>
      <td>192</td>
      <td>2000</td>
      <td>2000</td>
      <td>2008</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>



### Time for data cleaning.  

Columns with fewer datapoints than the total number of rows likely possess NaN values

Columns with only 1-2 missing row elements are the simplest case, someone probably messed up during data entry.  As it's only one element, try to set to a default/mean/most likely value, or an "other" category if there is one


```python
print(df['SaleType'][df['SaleType'].isnull()])
print(df['KitchenQual'][df['KitchenQual'].isnull()])
print(df['Electrical'][df['Electrical'].isnull()])
print(df['Exterior1st'][df['Exterior1st'].isnull()])
print(df['Exterior2nd'][df['Exterior2nd'].isnull()])
print(df['GarageArea'][df['GarageArea'].isnull()])
print(df['GarageCars'][df['GarageCars'].isnull()])
print(df['Utilities'][df['Utilities'].isnull()])
print(df['Functional'][df['Functional'].isnull()])
print(df['BsmtFullBath'][df['BsmtFullBath'].isnull()])
print(df['BsmtHalfBath'][df['BsmtHalfBath'].isnull()])

```

    1029    NaN
    Name: SaleType, dtype: object
    95    NaN
    Name: KitchenQual, dtype: object
    1379    NaN
    Name: Electrical, dtype: object
    691    NaN
    Name: Exterior1st, dtype: object
    691    NaN
    Name: Exterior2nd, dtype: object
    1116   NaN
    Name: GarageArea, dtype: float64
    1116   NaN
    Name: GarageCars, dtype: float64
    455    NaN
    485    NaN
    Name: Utilities, dtype: object
    756     NaN
    1013    NaN
    Name: Functional, dtype: object
    660   NaN
    728   NaN
    Name: BsmtFullBath, dtype: float64
    660   NaN
    728   NaN
    Name: BsmtHalfBath, dtype: float64



```python
df.Functional.value_counts()
```




    Typ     2717
    Min2      70
    Min1      65
    Mod       35
    Maj1      19
    Maj2       9
    Sev        2
    Name: Functional, dtype: int64



### Note: while going through this dataset I looked at each variable's distribution first before deciding to fill in N/A values with the mean or the most common value or the most "default" value.


To help do this, I made use of either the histogram, as seen in Sale Price, or the .value_counts() command above, or the .unique() command

For numerical variables one can do a mini-interpolation to fill in the blanks if there are a lot of N/A values


```python
df['SaleType'] = df['SaleType'].fillna("Oth")
df['KitchenQual'] = df['KitchenQual'].fillna('TA')
df['Electrical'] = df['Electrical'].fillna('SBrkr')
df['Exterior1st'] = df['Exterior1st'].fillna('VinylSd') #Other
df['Exterior2nd'] = df['Exterior2nd'].fillna('VinylSd')
df['GarageArea'] = df['GarageArea'].fillna(np.mean(df['GarageArea']))
df['GarageCars'] = df['GarageCars'].fillna(0)
df['Utilities'] = df['Utilities'].fillna('AllPub')
df['Functional'] = df['Functional'].fillna('Typ')
df['BsmtFullBath'] = df['BsmtFullBath'].fillna(0)
df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(0)




```


```python
print(df['Utilities'].unique())
print(df.Utilities.value_counts())


df['Utilities'][df['Utilities'] == 'NoSeWa']

# Thought to consider - can we get rid of 'Utilities'?
```

    ['AllPub' 'NoSeWa']
    AllPub    2918
    NoSeWa       1
    Name: Utilities, dtype: int64





    944    NoSeWa
    Name: Utilities, dtype: object



### For some others, it appears that the "NaN" values are intended to be a category amongst themselves.  

For these we substitute a new categorical result, "None"


```python
print(df['Alley'].unique())
df['Alley'] = df['Alley'].fillna('None')



```

    [nan 'Grvl' 'Pave']



```python

print(df['BsmtCond'].unique())
print(df['BsmtExposure'].unique())


df['BsmtCond'] = df['BsmtCond'].fillna('None')
df['BsmtExposure'] = df['BsmtExposure'].fillna('None')

#print(df['BsmtCond'].unique())
#print(df['BsmtExposure'].unique())


```

    ['TA' 'Gd' nan 'Fa' 'Po']
    ['No' 'Gd' 'Mn' 'Av' nan]



```python
df.MiscFeature.value_counts()
```




    Shed    95
    Gar2     5
    Othr     4
    TenC     1
    Name: MiscFeature, dtype: int64




```python
print(df['BsmtFinSF1'][df['BsmtFinSF1'].isnull()])
print(df['BsmtFinSF2'][df['BsmtFinSF2'].isnull()])
print(df['TotalBsmtSF'][df['TotalBsmtSF'].isnull()])
print(df['BsmtUnfSF'][df['BsmtUnfSF'].isnull()])


df['BsmtFinSF1'].fillna(np.mean(df['BsmtFinSF1'].dropna()), inplace=True)
df['BsmtFinSF2'].fillna(np.mean(df['BsmtFinSF2'].dropna()), inplace=True)
df['TotalBsmtSF'].fillna(np.mean(df['TotalBsmtSF'].dropna()), inplace=True)
df['BsmtUnfSF'].fillna(np.mean(df['BsmtUnfSF'].dropna()), inplace=True)



```

    660   NaN
    Name: BsmtFinSF1, dtype: float64
    660   NaN
    Name: BsmtFinSF2, dtype: float64
    660   NaN
    Name: TotalBsmtSF, dtype: float64
    660   NaN
    Name: BsmtUnfSF, dtype: float64



```python
df[['BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', 'BsmtUnfSF', 'SaleType', 'KitchenQual', \
   'Electrical', 'Exterior1st', 'Exterior2nd', 'GarageArea', 'GarageCars', 'Utilities', \
   'Functional', 'BsmtFullBath', 'BsmtHalfBath']].head(10)

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>TotalBsmtSF</th>
      <th>BsmtUnfSF</th>
      <th>SaleType</th>
      <th>KitchenQual</th>
      <th>Electrical</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>GarageArea</th>
      <th>GarageCars</th>
      <th>Utilities</th>
      <th>Functional</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>706.0</td>
      <td>0.0</td>
      <td>856.0</td>
      <td>150.0</td>
      <td>WD</td>
      <td>Gd</td>
      <td>SBrkr</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>548.0</td>
      <td>2.0</td>
      <td>AllPub</td>
      <td>Typ</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>978.0</td>
      <td>0.0</td>
      <td>1262.0</td>
      <td>284.0</td>
      <td>WD</td>
      <td>TA</td>
      <td>SBrkr</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>460.0</td>
      <td>2.0</td>
      <td>AllPub</td>
      <td>Typ</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>486.0</td>
      <td>0.0</td>
      <td>920.0</td>
      <td>434.0</td>
      <td>WD</td>
      <td>Gd</td>
      <td>SBrkr</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>608.0</td>
      <td>2.0</td>
      <td>AllPub</td>
      <td>Typ</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>216.0</td>
      <td>0.0</td>
      <td>756.0</td>
      <td>540.0</td>
      <td>WD</td>
      <td>Gd</td>
      <td>SBrkr</td>
      <td>Wd Sdng</td>
      <td>Wd Shng</td>
      <td>642.0</td>
      <td>3.0</td>
      <td>AllPub</td>
      <td>Typ</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>655.0</td>
      <td>0.0</td>
      <td>1145.0</td>
      <td>490.0</td>
      <td>WD</td>
      <td>Gd</td>
      <td>SBrkr</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>836.0</td>
      <td>3.0</td>
      <td>AllPub</td>
      <td>Typ</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>732.0</td>
      <td>0.0</td>
      <td>796.0</td>
      <td>64.0</td>
      <td>WD</td>
      <td>TA</td>
      <td>SBrkr</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>480.0</td>
      <td>2.0</td>
      <td>AllPub</td>
      <td>Typ</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1369.0</td>
      <td>0.0</td>
      <td>1686.0</td>
      <td>317.0</td>
      <td>WD</td>
      <td>Gd</td>
      <td>SBrkr</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>636.0</td>
      <td>2.0</td>
      <td>AllPub</td>
      <td>Typ</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>859.0</td>
      <td>32.0</td>
      <td>1107.0</td>
      <td>216.0</td>
      <td>WD</td>
      <td>TA</td>
      <td>SBrkr</td>
      <td>HdBoard</td>
      <td>HdBoard</td>
      <td>484.0</td>
      <td>2.0</td>
      <td>AllPub</td>
      <td>Typ</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>952.0</td>
      <td>952.0</td>
      <td>WD</td>
      <td>TA</td>
      <td>FuseF</td>
      <td>BrkFace</td>
      <td>Wd Shng</td>
      <td>468.0</td>
      <td>2.0</td>
      <td>AllPub</td>
      <td>Min1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>851.0</td>
      <td>0.0</td>
      <td>991.0</td>
      <td>140.0</td>
      <td>WD</td>
      <td>TA</td>
      <td>SBrkr</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>205.0</td>
      <td>1.0</td>
      <td>AllPub</td>
      <td>Typ</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df['BsmtFinType1'].unique()) # Here NaN means no basement
print(df['BsmtFinType2'].unique())

df['BsmtFinType1']=df['BsmtFinType1'].fillna('None')
df['BsmtFinType2']=df['BsmtFinType2'].fillna('None')

```

    ['GLQ' 'ALQ' 'Unf' 'Rec' 'BLQ' nan 'LwQ']
    ['Unf' 'BLQ' nan 'ALQ' 'Rec' 'LwQ' 'GLQ']



```python
df['BsmtQual'] = df['BsmtQual'].fillna('None')
df['Fence'] = df['Fence'].fillna('None')

```


```python
print(df.FireplaceQu.value_counts())
df['FireplaceQu']= df['FireplaceQu'].fillna('None')


```

    Gd    744
    TA    592
    Fa     74
    Po     46
    Ex     43
    Name: FireplaceQu, dtype: int64



```python
df.Fireplaces.value_counts()
```




    0    1420
    1    1268
    2     219
    3      11
    4       1
    Name: Fireplaces, dtype: int64



### Sanity check: this XOR should return an empty dataframe, because the number of fireplaces should be 0 at the same time the fireplace quality is nothing



```python
df[(df['FireplaceQu']== 'None') ^ (df['Fireplaces'] ==0)]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>3SsnPorch</th>
      <th>Alley</th>
      <th>BedroomAbvGr</th>
      <th>BldgType</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>...</th>
      <th>SaleType</th>
      <th>ScreenPorch</th>
      <th>Street</th>
      <th>TotRmsAbvGrd</th>
      <th>TotalBsmtSF</th>
      <th>Utilities</th>
      <th>WoodDeckSF</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>YrSold</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 81 columns</p>
</div>



### Next, what to do with the year the garage was built, if the house does not have a garage?

NaN values will confuse a retrieval algorithm. For now, the simple mean of the years is substituted in, as a neutral value.


```python
print(df['GarageCond'].unique())
print(df['GarageFinish'].unique())
print(df['GarageQual'].unique())
print(df['GarageType'].unique())
print(df['GarageYrBlt'].unique())

df['GarageCond'].fillna('None', inplace=True)
df['GarageFinish'].fillna('None', inplace=True)
df['GarageQual'].fillna('None', inplace=True)
df['GarageType'].fillna('None', inplace=True)
df['GarageYrBlt'].fillna(np.mean(df['GarageYrBlt']), inplace=True)

```

    ['TA' 'Fa' nan 'Gd' 'Po' 'Ex']
    ['RFn' 'Unf' 'Fin' nan]
    ['TA' 'Fa' 'Gd' nan 'Ex' 'Po']
    ['Attchd' 'Detchd' 'BuiltIn' 'CarPort' nan 'Basment' '2Types']
    [ 2003.  1976.  2001.  1998.  2000.  1993.  2004.  1973.  1931.  1939.
      1965.  2005.  1962.  2006.  1960.  1991.  1970.  1967.  1958.  1930.
      2002.  1968.  2007.  2008.  1957.  1920.  1966.  1959.  1995.  1954.
      1953.    nan  1983.  1977.  1997.  1985.  1963.  1981.  1964.  1999.
      1935.  1990.  1945.  1987.  1989.  1915.  1956.  1948.  1974.  2009.
      1950.  1961.  1921.  1900.  1979.  1951.  1969.  1936.  1975.  1971.
      1923.  1984.  1926.  1955.  1986.  1988.  1916.  1932.  1972.  1918.
      1980.  1924.  1996.  1940.  1949.  1994.  1910.  1978.  1982.  1992.
      1925.  1941.  2010.  1927.  1947.  1937.  1942.  1938.  1952.  1928.
      1922.  1934.  1906.  1914.  1946.  1908.  1929.  1933.  1917.  1896.
      1895.  2207.  1943.  1919.]



```python
df[['LotArea', 'LotConfig', 'LotFrontage', 'LotShape']][df['LotFrontage'].isnull()]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LotArea</th>
      <th>LotConfig</th>
      <th>LotFrontage</th>
      <th>LotShape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7</th>
      <td>10382</td>
      <td>Corner</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12968</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>10920</td>
      <td>Corner</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>11241</td>
      <td>CulDSac</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>8246</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>31</th>
      <td>8544</td>
      <td>CulDSac</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>42</th>
      <td>9180</td>
      <td>CulDSac</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>43</th>
      <td>9200</td>
      <td>CulDSac</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>50</th>
      <td>13869</td>
      <td>Corner</td>
      <td>NaN</td>
      <td>IR2</td>
    </tr>
    <tr>
      <th>64</th>
      <td>9375</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>66</th>
      <td>19900</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>76</th>
      <td>8475</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>84</th>
      <td>8530</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>95</th>
      <td>9765</td>
      <td>Corner</td>
      <td>NaN</td>
      <td>IR2</td>
    </tr>
    <tr>
      <th>100</th>
      <td>10603</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>104</th>
      <td>7758</td>
      <td>Corner</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>111</th>
      <td>7750</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>113</th>
      <td>21000</td>
      <td>Corner</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>116</th>
      <td>11616</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>120</th>
      <td>21453</td>
      <td>CulDSac</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>126</th>
      <td>4928</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>131</th>
      <td>12224</td>
      <td>Corner</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>133</th>
      <td>6853</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>136</th>
      <td>10355</td>
      <td>Corner</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>147</th>
      <td>9505</td>
      <td>CulDSac</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>149</th>
      <td>6240</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>152</th>
      <td>14803</td>
      <td>CulDSac</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>153</th>
      <td>13500</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>160</th>
      <td>11120</td>
      <td>CulDSac</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>166</th>
      <td>10708</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1224</th>
      <td>12585</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1240</th>
      <td>9019</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1243</th>
      <td>9240</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1244</th>
      <td>9308</td>
      <td>CulDSac</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1246</th>
      <td>8638</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1247</th>
      <td>13052</td>
      <td>CulDSac</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1248</th>
      <td>8020</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1249</th>
      <td>8789</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1254</th>
      <td>2998</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>1255</th>
      <td>4447</td>
      <td>FR2</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1264</th>
      <td>9759</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1267</th>
      <td>10368</td>
      <td>CulDSac</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1277</th>
      <td>8917</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1278</th>
      <td>12700</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1281</th>
      <td>9610</td>
      <td>Corner</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1304</th>
      <td>18275</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1347</th>
      <td>11327</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1350</th>
      <td>9535</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1351</th>
      <td>7176</td>
      <td>CulDSac</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1352</th>
      <td>9662</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1354</th>
      <td>17529</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>1355</th>
      <td>20355</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>1358</th>
      <td>1700</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>1379</th>
      <td>8685</td>
      <td>CulDSac</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1385</th>
      <td>9930</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1387</th>
      <td>11088</td>
      <td>Corner</td>
      <td>NaN</td>
      <td>Reg</td>
    </tr>
    <tr>
      <th>1390</th>
      <td>21533</td>
      <td>FR2</td>
      <td>NaN</td>
      <td>IR2</td>
    </tr>
    <tr>
      <th>1440</th>
      <td>50102</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1441</th>
      <td>8098</td>
      <td>Inside</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
    <tr>
      <th>1448</th>
      <td>11836</td>
      <td>Corner</td>
      <td>NaN</td>
      <td>IR1</td>
    </tr>
  </tbody>
</table>
<p>486 rows × 4 columns</p>
</div>



### There isn't quite a clear explanation for why "Lot Frontage" has NaN values, so here the mean will be substituted in.


```python
print(df['LotFrontage'].unique())

df['LotFrontage'].hist(bins=20)

df['LotFrontage'].fillna(np.mean(df['LotFrontage']), inplace=True)


```

    [  65.   80.   68.   60.   84.   85.   75.   nan   51.   50.   70.   91.
       72.   66.  101.   57.   44.  110.   98.   47.  108.  112.   74.  115.
       61.   48.   33.   52.  100.   24.   89.   63.   76.   81.   95.   69.
       21.   32.   78.  121.  122.   40.  105.   73.   77.   64.   94.   34.
       90.   55.   88.   82.   71.  120.  107.   92.  134.   62.   86.  141.
       97.   54.   41.   79.  174.   99.   67.   83.   43.  103.   93.   30.
      129.  140.   35.   37.  118.   87.  116.  150.  111.   49.   96.   59.
       36.   56.  102.   58.   38.  109.  130.   53.  137.   45.  106.  104.
       42.   39.  144.  114.  128.  149.  313.  168.  182.  138.  160.  152.
      124.  153.   46.   26.   25.  119.   31.   28.  117.  113.  125.  135.
      136.   22.  123.  195.  155.  126.  200.  131.  133.]



![png](House%20Price%20Prediction_files/House%20Price%20Prediction_32_1.png)


For some cases marking as "unknown" is fine.  Marking something as unknown is far less satisfying.

It's not quite clear, for example, with Masonry Veneer Type ('MasVnrType') whether the NaNs and Nones are the same as both exist in the dataset as is; it's only our educated best guess that they're the same

For Misc. Features there's an "other" category and a "None" category.  They're technically distinct but in this case "other" is vague enough that we will link them together


```python
print(df.MSZoning.value_counts())
print(df.MiscFeature.value_counts())

df['MSZoning'].fillna('RL', inplace=True)   # RL most common
df['MasVnrArea'].fillna(0,inplace=True) #fill in median np.mean(df['MasVnrArea']), inplace=True)  
df['MasVnrType'].fillna('None', inplace=True)  
df['MiscFeature'].fillna('Othr', inplace=True)  
df['PoolQC'].fillna('No Pool', inplace=True)  

```

    RL         2265
    RM          460
    FV          139
    RH           26
    C (all)      25
    Name: MSZoning, dtype: int64
    Shed    95
    Gar2     5
    Othr     4
    TenC     1
    Name: MiscFeature, dtype: int64



```python
df['CentralAir'] = df['CentralAir'].map({'Y': 1, 'N':0})
```

### Here, note that 'MSSubClass' shoud be a categorical variable


```python
#Strangely this code reduces my accuracy.  Is information being lost, e.g. this is listed in an order of some kind?



df['MSSubClass']=df['MSSubClass'].map({20:'1-STORY 1946 & NEWER ALL STYLES', 30: '1-STORY 1945 & OLDER', 40: '1-STORY W/FINISHED ATTIC ALL AGES', 45: '1-1/2 STORY - UNFINISHED ALL AGES', 50: '1-1/2 STORY FINISHED ALL AGES', 60: '2-STORY 1946 & NEWER', 70: '2-STORY 1945 & OLDER', 75: '2-1/2 STORY ALL AGES', 80: 'SPLIT OR MULTI-LEVEL', 85: 'SPLIT FOYER', 90: 'DUPLEX - ALL STYLES AND AGES', 120: '1-STORY PUD (Planned Unit Development) - 1946 & NEWER', 150: '1-1/2 STORY PUD - ALL AGES', 160: '2-STORY PUD - 1946 & NEWER', 180: 'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER', 190: '2 FAMILY CONVERSION - ALL STYLES AND AGES'})


```


```python
df.MSSubClass.value_counts()
```




    1-STORY 1946 & NEWER ALL STYLES                          1079
    2-STORY 1946 & NEWER                                      575
    1-1/2 STORY FINISHED ALL AGES                             287
    1-STORY PUD (Planned Unit Development) - 1946 & NEWER     182
    1-STORY 1945 & OLDER                                      139
    2-STORY PUD - 1946 & NEWER                                128
    2-STORY 1945 & OLDER                                      128
    SPLIT OR MULTI-LEVEL                                      118
    DUPLEX - ALL STYLES AND AGES                              109
    2 FAMILY CONVERSION - ALL STYLES AND AGES                  61
    SPLIT FOYER                                                48
    2-1/2 STORY ALL AGES                                       23
    1-1/2 STORY - UNFINISHED ALL AGES                          18
    PUD - MULTILEVEL - INCL SPLIT LEV/FOYER                    17
    1-STORY W/FINISHED ATTIC ALL AGES                           6
    1-1/2 STORY PUD - ALL AGES                                  1
    Name: MSSubClass, dtype: int64




```python
df['GarageType'].unique()
```




    array(['Attchd', 'Detchd', 'BuiltIn', 'CarPort', 'None', 'Basment',
           '2Types'], dtype=object)




```python
df['Utilities'].unique()


```




    array(['AllPub', 'NoSeWa'], dtype=object)




```python
fig = plt.figure(figsize= (10,10))

sns.heatmap(df.corr())
```

    /Users/rosscheung/anaconda/lib/python3.4/site-packages/matplotlib/collections.py:590: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if self._edgecolors == str('face'):





    <matplotlib.axes._subplots.AxesSubplot at 0x13fa52550>




![png](House%20Price%20Prediction_files/House%20Price%20Prediction_41_2.png)


### Let's plot some of the data that's more strongly correlated with the Sale Price


```python

fig = plt.figure(figsize= (10,10))
ax = fig.add_subplot(221)
ax.plot(df['GrLivArea'], df['SalePrice'], '.') # Note that there is a long tail; someone makes 8 million a year
ax.set_title('Above grade (ground) living area square feet')


ax = fig.add_subplot(222)
ax.plot(df['BsmtFinSF1'], df['SalePrice'], '.')
ax.set_title('Type 1 finished square feet')

ax = fig.add_subplot(223)
ax.plot(df['TotalBsmtSF'], df['SalePrice'], '.')
ax.set_title('Total square feet of basement area')


ax = fig.add_subplot(224)
ax.plot(df['GarageArea'], df['SalePrice'], '.')
ax.set_title('Size of garage in square feet')


```




    <matplotlib.text.Text at 0x1406436a0>




![png](House%20Price%20Prediction_files/House%20Price%20Prediction_43_1.png)


### It's worth noting that there are two datapoints with large living area but low prices, which buck the trends

Deciding what to remove as outliers is always a tricky matter, which should be done sparingly and with thought.  But later (after splitting the train/test sets) we will do it.


```python
df[df['GrLivArea']>4000][['SalePrice', 'GrLivArea', 'BsmtFinSF1', 'TotalBsmtSF', 'GarageArea']]

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SalePrice</th>
      <th>GrLivArea</th>
      <th>BsmtFinSF1</th>
      <th>TotalBsmtSF</th>
      <th>GarageArea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>523</th>
      <td>184750.0</td>
      <td>4676</td>
      <td>2260.0</td>
      <td>3138.0</td>
      <td>884.0</td>
    </tr>
    <tr>
      <th>691</th>
      <td>755000.0</td>
      <td>4316</td>
      <td>1455.0</td>
      <td>2444.0</td>
      <td>832.0</td>
    </tr>
    <tr>
      <th>1182</th>
      <td>745000.0</td>
      <td>4476</td>
      <td>2096.0</td>
      <td>2396.0</td>
      <td>813.0</td>
    </tr>
    <tr>
      <th>1298</th>
      <td>160000.0</td>
      <td>5642</td>
      <td>5644.0</td>
      <td>6110.0</td>
      <td>1418.0</td>
    </tr>
    <tr>
      <th>1089</th>
      <td>NaN</td>
      <td>5095</td>
      <td>4010.0</td>
      <td>5095.0</td>
      <td>1154.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.groupby('MoSold')['SalePrice'].mean())

plt.bar(np.arange(1,13), df.groupby('MoSold')['SalePrice'].mean(), )

df['MoSold'] = df['MoSold'].map({1: 'January', 2: 'February', 3: 'March', 4:'April', 5:'May', 6:'June', 7:'July',8:'August', 9:'September', 10:'October',11:'November',12:'December'})


```

    MoSold
    1     183256.258621
    2     177882.000000
    3     183253.924528
    4     171503.262411
    5     172307.269608
    6     177395.735178
    7     186331.192308
    8     184651.827869
    9     195683.206349
    10    179563.977528
    11    192210.911392
    12    186518.966102
    Name: SalePrice, dtype: float64



![png](House%20Price%20Prediction_files/House%20Price%20Prediction_46_1.png)


### Feature Engineering midwesterners living the American dream

There appears to be two values where the living area is greater than 4000, but the sale price is super low.  We may consider removing these as outliers much later (after splitting up the training and test sets)

By domain knowledge (thanks Michele!), a "4 bed-2 bath" house is especially prized.  Let's consider this special category of a house and flag it.


```python
print(df['BedroomAbvGr'].unique())

print(df['FullBath'].unique())


df_4bed_2bath = df[(df['BedroomAbvGr'] ==4) & (df['FullBath']==2)]
df_everthingelse = df[~((df['BedroomAbvGr'] ==4) & (df['FullBath']==2))]



df_4bed_2bath.head()

fig = plt.figure(figsize= (10,10))
ax = fig.add_subplot(211)
ax.hist(df_everthingelse['SalePrice'].dropna(), 100)
ax.set_xlim([0, 500000])
ax = fig.add_subplot(212)
ax.hist(df_4bed_2bath['SalePrice'].dropna().values, 100)
ax.set_xlim([0, 500000])


df_4bed_2bath['SalePrice'].mean(), df_everthingelse['SalePrice'].mean()
```

    [3 4 1 2 0 5 6 8]
    [2 1 3 0 4]





    (212340.12820512822, 177162.48926380367)




![png](House%20Price%20Prediction_files/House%20Price%20Prediction_48_2.png)


When implemented, this does improve score very slightly (0.00014)



```python

df['4bed2bath'] = 0
df['4bed2bath']= (df['BedroomAbvGr'] ==4) & (df['FullBath']==2)

```

## Can we create a new variable, floor space, out of existing ones?

For those of you doing the Titanic challenge, a similar trick can be done to create a "total family size" variable out of siblings and married couples, etc.  It in fact adds more information.  




```python

df['TotalFloorSpace'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
```

## Neighborhoods

For a lot of homeowners the choice of a neighborhood may influence buying houses for a number of factors, including the fact that neighborhood is part of people's identity and outward appearance.

(note: the author grew up in an affluent suburb and has some familiarity with how yuppies think)


```python
df['Neighborhood'].unique()
```




    array(['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst',
           'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes',
           'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert',
           'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU',
           'Blueste'], dtype=object)




```python
neighborhood_price = df.groupby('Neighborhood')['SalePrice'].mean()

neighborhood_floorspace = df.groupby('Neighborhood')['GrLivArea'].mean()
print(neighborhood_price)

neighborhood_price.sort_values().plot(kind='bar')

```

    Neighborhood
    Blmngtn    194870.882353
    Blueste    137500.000000
    BrDale     104493.750000
    BrkSide    124834.051724
    ClearCr    212565.428571
    CollgCr    197965.773333
    Crawfor    210624.725490
    Edwards    128219.700000
    Gilbert    192854.506329
    IDOTRR     100123.783784
    MeadowV     98576.470588
    Mitchel    156270.122449
    NAmes      145847.080000
    NPkVill    142694.444444
    NWAmes     189050.068493
    NoRidge    335295.317073
    NridgHt    316270.623377
    OldTown    128225.300885
    SWISU      142591.360000
    Sawyer     136793.135135
    SawyerW    186555.796610
    Somerst    225379.837209
    StoneBr    310499.000000
    Timber     242247.447368
    Veenker    238772.727273
    Name: SalePrice, dtype: float64





    <matplotlib.axes._subplots.AxesSubplot at 0x14071a710>




![png](House%20Price%20Prediction_files/House%20Price%20Prediction_55_2.png)



```python
neighborhood_floorspace.sort_values().plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1410c0be0>




![png](House%20Price%20Prediction_files/House%20Price%20Prediction_56_1.png)


Places like New York City and San Francisco are testimonials that floor space and wealthy areas aren't necessarily correlated.  However price, tempered by floor space is a useful metric for determining this.

Admittedly in creating a binary variable here, I just "eyeballed" a cutoff.  Perhaps a more robust method can be determined in the future


```python
# rich places: NoRidge, NridgeHt, StoneBr,

rich_neighborhoods = ['NoRidge', 'StoneBr', 'NridgHt', 'Veenker', 'ClearCr', 'Crawfor', 'Timber','Somerst']

df['RichNeighborhood'] = df['Neighborhood'].isin(rich_neighborhoods) * 1

df[['Neighborhood', 'RichNeighborhood']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>RichNeighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CollgCr</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Veenker</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CollgCr</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Crawfor</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NoRidge</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Mitchel</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Somerst</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NWAmes</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>OldTown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BrkSide</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Sawyer</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>NridgHt</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Sawyer</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>CollgCr</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>NAmes</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>BrkSide</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>NAmes</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Sawyer</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>SawyerW</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>NAmes</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>NridgHt</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>IDOTRR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>CollgCr</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>MeadowV</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Sawyer</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>NridgHt</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>NAmes</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>NridgHt</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>NAmes</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>BrkSide</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1429</th>
      <td>IDOTRR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1430</th>
      <td>IDOTRR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1431</th>
      <td>IDOTRR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1432</th>
      <td>IDOTRR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1433</th>
      <td>IDOTRR</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1434</th>
      <td>Crawfor</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1435</th>
      <td>Crawfor</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1436</th>
      <td>Mitchel</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1437</th>
      <td>Mitchel</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1438</th>
      <td>Mitchel</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1439</th>
      <td>Mitchel</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1440</th>
      <td>Timber</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1441</th>
      <td>Timber</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1442</th>
      <td>Timber</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1443</th>
      <td>Timber</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1444</th>
      <td>Mitchel</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1445</th>
      <td>Mitchel</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1446</th>
      <td>MeadowV</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1447</th>
      <td>Mitchel</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1448</th>
      <td>Mitchel</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1449</th>
      <td>MeadowV</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1450</th>
      <td>MeadowV</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1451</th>
      <td>Mitchel</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1452</th>
      <td>MeadowV</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1453</th>
      <td>MeadowV</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1454</th>
      <td>MeadowV</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>MeadowV</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>Mitchel</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>Mitchel</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>Mitchel</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2919 rows × 2 columns</p>
</div>




```python
neighborhood_floorspace
```




    Neighborhood
    Blmngtn    1404.892857
    Blueste    1159.700000
    BrDale     1115.233333
    BrkSide    1234.907407
    ClearCr    1744.386364
    CollgCr    1496.119850
    Crawfor    1722.796117
    Edwards    1337.737113
    Gilbert    1620.896970
    IDOTRR     1205.247312
    MeadowV    1066.702703
    Mitchel    1327.991228
    NAmes      1292.054176
    NPkVill    1244.086957
    NWAmes     1688.770992
    NoRidge    2480.633803
    NridgHt    1942.638554
    OldTown    1431.974895
    SWISU      1636.479167
    Sawyer     1183.026490
    SawyerW    1604.064000
    Somerst    1604.829670
    StoneBr    1949.215686
    Timber     1714.638889
    Veenker    1819.541667
    Name: GrLivArea, dtype: float64




```python
df['Neighborhood'].head(10)
```




    0    CollgCr
    1    Veenker
    2    CollgCr
    3    Crawfor
    4    NoRidge
    5    Mitchel
    6    Somerst
    7     NWAmes
    8    OldTown
    9    BrkSide
    Name: Neighborhood, dtype: object



## Difference in house floor space from neighbords

Let's create a new variable that is the ratio between a home, and the mean floor space of the neighborhood, assuming people don't want to be stuck with the smallest home in a neighborhood.  One has to show off to ones neighbors after all.


```python
df['SizeDifferenceFromNeighbors'] = df['TotalFloorSpace'] / df['Neighborhood'].map(neighborhood_floorspace)

plt.scatter( df['SizeDifferenceFromNeighbors'],df['SalePrice'])
```




    <matplotlib.collections.PathCollection at 0x1407557f0>



    /Users/rosscheung/anaconda/lib/python3.4/site-packages/matplotlib/collections.py:590: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if self._edgecolors == str('face'):



![png](House%20Price%20Prediction_files/House%20Price%20Prediction_62_2.png)


## What about bedrooms and bathrooms?

If there are not enough bedrooms and bathrooms, especially with kids, people may be unhappy


```python
df['numBathrooms'] = df['BsmtFullBath'] + df['FullBath'] + df['BsmtHalfBath']
# Question: what about basement half bathrooms?

df['Bathroom-Bedroom ratio'] = df['numBathrooms']/ (df['BedroomAbvGr']+1)

# There doesn't seem to be any improvement if this ratio is higher than 1 (house occupants can't use more than 1 at a  time)
plt.scatter(df['Bathroom-Bedroom ratio'], df['SalePrice'])

```




    <matplotlib.collections.PathCollection at 0x13fad84e0>



    /Users/rosscheung/anaconda/lib/python3.4/site-packages/matplotlib/collections.py:590: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if self._edgecolors == str('face'):



![png](House%20Price%20Prediction_files/House%20Price%20Prediction_64_2.png)



```python
fig = plt.figure(figsize= (10,10))
ax = fig.add_subplot(321)
ax.plot(df['FullBath'], df['SalePrice'], '.')
ax.set_title('Number of Bathrooms')


ax = fig.add_subplot(322)
ax.plot(df['HalfBath'], df['SalePrice'], '.')
ax.set_title('Number of Half Bathrooms')


ax = fig.add_subplot(323)
ax.plot(df['numBathrooms'], df['SalePrice'], '.')
ax.set_title('Total Bathrooms')


ax = fig.add_subplot(324)
ax.plot(df['BedroomAbvGr'], df['SalePrice'], '.')
ax.set_title('Above ground Bedrooms')


ax = fig.add_subplot(325)
ax.plot(df['BedroomAbvGr'], df['SalePrice'], '.')
ax.set_title('Above ground Bedrooms')

df['BathroomSq'] = df['FullBath'] * df['FullBath']
df['TotalBathroomSq'] = df['numBathrooms'] * df['numBathrooms']

```


![png](House%20Price%20Prediction_files/House%20Price%20Prediction_65_0.png)



```python
#df['KitchenQual'] = df['KitchenQual'].map({'Ex':5, 'Gd':4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
#df['FireplaceQu']= df['FireplaceQu'].map({'Ex':5, 'Gd':4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
#df['GarageQual']= df['GarageQual'].map({'Ex':5, 'Gd':4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
#df['GarageCond']= df['GarageCond'].map({'Ex':5, 'Gd':4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
#df['HeatingQC'] = df['HeatingQC'].map({'Ex':5, 'Gd':4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
#df['PoolQC'] = df['PoolQC'].map({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'No Pool': 0})
#df['BsmtCond'] = df['BsmtCond'].map({'Ex':5, 'Gd':4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
#df['ExterCond']= df['ExterCond'].map({'Ex':5, 'Gd':4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})
#df['ExterQual']= df['ExterQual'].map({'Ex':5, 'Gd':4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0})


```


```python
#df['BsmtFinType2'].map({'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2,'Unf': 1, 'None': 0})
#df['BsmtFinType2']
#df['BsmtFinType1']
#df['Fence'].unique()
#df['GarageFinish'] = df['GarageFinish'].map({'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0})

#sns.barplot(x= 'Fence', y = 'SalePrice', data = df_train)

fig = plt.figure(figsize= (10,10))
ax = fig.add_subplot(321)

plt.plot(df['OverallCond'], df['SalePrice'], '.')

ax = fig.add_subplot(322)
plt.plot(df['YrSold'], df['SalePrice'], '.')

#ax = fig.add_subplot(223)
#plt.plot(df['Functional'], df['SalePrice'], '.')
ax = fig.add_subplot(323)
plt.plot(df['GrLivArea'], df['SalePrice'], '.')


ax = fig.add_subplot(324)
plt.plot(df['LotFrontage'], df['SalePrice'], '.')


#plt.plot(df['LowQualFinSF'], df['SalePrice'], '.')

ax = fig.add_subplot(325)
plt.plot(df['LotArea'], df['SalePrice'], '.')
ax = fig.add_subplot(326)
plt.plot(np.log1p(df['LotArea']), df['SalePrice'], '.')


```




    [<matplotlib.lines.Line2D at 0x1382315c0>]




![png](House%20Price%20Prediction_files/House%20Price%20Prediction_67_1.png)



```python
df['YrSold'] = df['YrSold'].astype('str')

```


```python
plt.figure(1)
plt.plot(df_train['YearBuilt'], df_train['SalePrice'], '.')

#plt.figure(1)
#plt.plot(df_train['YearRemodAdd'], df_train['SalePrice'], '.r')

```




    [<matplotlib.lines.Line2D at 0x13f2fa8d0>]




![png](House%20Price%20Prediction_files/House%20Price%20Prediction_69_1.png)



```python

#Heating
#plt.plot( df_train['SalePrice'],df_train['SaleType'], kind='bar')
```


```python
#df['YearRemodAdd']  = 2017 - df['YearRemodAdd']
#df['YearBuilt'] = 2017 - df['YearBuilt']

print(df['Condition1'].unique())
print(df['Condition2'].unique())
```

    ['Norm' 'Feedr' 'PosN' 'Artery' 'RRAe' 'RRNn' 'RRAn' 'PosA' 'RRNe']
    ['Norm' 'Artery' 'RRNn' 'Feedr' 'PosN' 'PosA' 'RRAn' 'RRAe']



```python

df['Location_Norm']=  (df['Condition1'] == 'Norm') | (df['Condition2'] == 'Norm')
df['Location_Feedr'] = ((df['Condition1'] == 'Feedr') | (df['Condition2'] == 'Feedr'))
df['Location_Artery'] = ((df['Condition1'] == 'Artery') | (df['Condition2'] == 'Artery'))
df['Location_RRNn'] = ((df['Condition1'] == 'RRNn') | (df['Condition2'] == 'RRNn'))
df['Location_RRAn'] = ((df['Condition1'] == 'RRAn') | (df['Condition2'] == 'RRAn'))
df['Location_PosN'] = ((df['Condition1'] == 'PosN') | (df['Condition2'] == 'PosN'))
df['Location_PosA'] = ((df['Condition1'] == 'PosA') | (df['Condition2'] == 'PosA'))
df['Location_RRNe'] = ((df['Condition1'] == 'RRNe') | (df['Condition2'] == 'RRNe'))
df['Location_RRAe'] = ((df['Condition1'] == 'RRAe') | (df['Condition2'] == 'RRAe'))


df['Location_Norm'] = df['Location_Norm'].map(lambda x: 1 if x else 0)
df['Location_Feedr'] = df['Location_Feedr'].map(lambda x: 1 if x else 0)
df['Location_Artery'] = df['Location_Artery'].map(lambda x: 1 if x else 0)
df['Location_RRNn'] = df['Location_RRNn'].map(lambda x: 1 if x else 0)
df['Location_RRAn'] = df['Location_RRAn'].map(lambda x: 1 if x else 0)
df['Location_PosN'] = df['Location_PosN'].map(lambda x: 1 if x else 0)
df['Location_PosA'] = df['Location_PosA'].map(lambda x: 1 if x else 0)
df['Location_RRNe'] = df['Location_RRNe'].map(lambda x: 1 if x else 0)
df['Location_RRAe'] = df['Location_RRAe'].map(lambda x: 1 if x else 0)


df.drop(['Condition1',  'Condition2'], axis=1, inplace=True)

```


```python
df['Exterior1st'].unique()
```




    array(['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing',
           'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'BrkComm', 'AsphShn',
           'Stone', 'ImStucc', 'CBlock'], dtype=object)



Note: to convert a boolean (True/False) to a number (0 or 1), another way is to just do an arithmetic option like adding 0 or multiplying by 1.  Then Python will just convert the boolean to a numeric.


```python
#Repeat for exterior 1st and 2nd?
df['Exterior_AsbShng']=  ((df['Exterior1st'] == 'AsbShng') | (df['Exterior2nd'] == 'AsbShng')) + 0
df['Exterior_AsphShn']=  ((df['Exterior1st'] == 'AsphShn') | (df['Exterior2nd'] == 'AsphShn')) + 0
df['Exterior_BrkComm']=  ((df['Exterior1st'] == 'BrkComm') | (df['Exterior2nd'] == 'BrkComm')) + 0
df['Exterior_BrkFace']=  ((df['Exterior1st'] == 'BrkFace') | (df['Exterior2nd'] == 'BrkFace')) + 0
df['Exterior_CBlock']=  ((df['Exterior1st'] == 'CBlock') | (df['Exterior2nd'] == 'CBlock')) + 0
df['Exterior_CemntBd']=  ((df['Exterior1st'] == 'CemntBd') | (df['Exterior2nd'] == 'CemntBd')) + 0
df['Exterior_HdBoard']=  ((df['Exterior1st'] == 'HdBoard') | (df['Exterior2nd'] == 'HdBoard')) + 0
df['Exterior_ImStucc']=  ((df['Exterior1st'] == 'ImStucc') | (df['Exterior2nd'] == 'ImStucc')) + 0
df['Exterior_MetalSd']=  ((df['Exterior1st'] == 'MetalSd') | (df['Exterior2nd'] == 'MetalSd')) + 0
df['Exterior_Other']=  ((df['Exterior1st'] == 'Other') | (df['Exterior2nd'] == 'Other')) + 0
df['Exterior_Plywood']=  ((df['Exterior1st'] == 'Plywood') | (df['Exterior2nd'] == 'Plywood')) + 0
df['Exterior_PreCast']=  ((df['Exterior1st'] == 'PreCast') | (df['Exterior2nd'] == 'PreCast')) + 0
df['Exterior_Stone']=  ((df['Exterior1st'] == 'Stone') | (df['Exterior2nd'] == 'Stone')) + 0
df['Exterior_Stucco']=  ((df['Exterior1st'] == 'Stucco') | (df['Exterior2nd'] == 'Stucco')) + 0
df['Exterior_VinylSd']=  ((df['Exterior1st'] == 'VinylSd') | (df['Exterior2nd'] == 'VinylSd')) + 0
df['Exterior_Wd_Sdng']=  ((df['Exterior1st'] == 'Wd Sdng') | (df['Exterior2nd'] == 'Wd Sdng')) + 0
df['Exterior_WdShing']=  ((df['Exterior1st'] == 'WdShing') | (df['Exterior2nd'] == 'WdShing')) + 0


#df[['Exterior1st',  'Exterior2nd', 'Exterior_VinylSd', 'Exterior_AsphShn','Exterior_BrkComm']]
```


```python
df.drop(['Exterior1st',  'Exterior2nd'], axis=1, inplace=True)
```


```python
df['BsmtFinType2'].unique()
```




    array(['Unf', 'BLQ', 'None', 'ALQ', 'Rec', 'LwQ', 'GLQ'], dtype=object)




```python
df['BsmtFinType_GLQ']=  ((df['BsmtFinType1'] == 'GLQ') | (df['BsmtFinType2'] == 'GLQ')) + 0
df['BsmtFinType_ALQ']=  ((df['BsmtFinType1'] == 'ALQ') | (df['BsmtFinType2'] == 'ALQ')) + 0
df['BsmtFinType_BLQ']=  ((df['BsmtFinType1'] == 'BLQ') | (df['BsmtFinType2'] == 'BLQ')) + 0
df['BsmtFinType_Rec']=  ((df['BsmtFinType1'] == 'Rec') | (df['BsmtFinType2'] == 'Rec')) + 0
df['BsmtFinType_LwQ']=  ((df['BsmtFinType1'] == 'LwQ') | (df['BsmtFinType2'] == 'LwQ')) + 0
df['BsmtFinType_Unf']=  ((df['BsmtFinType1'] == 'Unf') | (df['BsmtFinType2'] == 'Unf')) + 0
df['BsmtFinType_None']=  ((df['BsmtFinType1'] == 'None') | (df['BsmtFinType2'] == 'None')) + 0


```


```python
df.drop(['BsmtFinType1',  'BsmtFinType2'], axis=1, inplace=True)
```

## Experiment with dropping rows with next to no information


```python
correlations= df.corr()['SalePrice'].sort_values(ascending=False )
print(correlations)
```

    SalePrice                      1.000000
    OverallQual                    0.790982
    TotalFloorSpace                0.782260
    GrLivArea                      0.708624
    GarageCars                     0.640409
    GarageArea                     0.623431
    TotalBsmtSF                    0.613581
    1stFlrSF                       0.605852
    RichNeighborhood               0.585580
    BathroomSq                     0.577302
    numBathrooms                   0.574226
    TotalBathroomSq                0.561809
    FullBath                       0.560664
    TotRmsAbvGrd                   0.533723
    YearBuilt                      0.522897
    YearRemodAdd                   0.507101
    MasVnrArea                     0.472614
    GarageYrBlt                    0.471062
    Fireplaces                     0.466929
    SizeDifferenceFromNeighbors    0.436532
    BsmtFinType_GLQ                0.430415
    Bathroom-Bedroom ratio         0.408142
    BsmtFinSF1                     0.386420
    LotFrontage                    0.334820
    WoodDeckSF                     0.324413
    2ndFlrSF                       0.319334
    OpenPorchSF                    0.315856
    Exterior_VinylSd               0.302553
    HalfBath                       0.284108
    LotArea                        0.263843
                                     ...   
    Location_RRAn                  0.002967
    Exterior_Stone                -0.000998
    Location_RRNn                 -0.001367
    BsmtFinSF2                    -0.011378
    BsmtHalfBath                  -0.016844
    MiscVal                       -0.021190
    Id                            -0.021917
    Exterior_AsphShn              -0.024524
    Exterior_CBlock               -0.025028
    LowQualFinSF                  -0.025606
    Exterior_Stucco               -0.038470
    Location_RRAe                 -0.043813
    Exterior_BrkComm              -0.051264
    Exterior_WdShing              -0.051317
    Exterior_Plywood              -0.054547
    OverallCond                   -0.077856
    BsmtFinType_ALQ               -0.086367
    Exterior_HdBoard              -0.091262
    BsmtFinType_LwQ               -0.091463
    Location_Artery               -0.106401
    Exterior_AsbShng              -0.109712
    Location_Feedr                -0.123694
    EnclosedPorch                 -0.128578
    KitchenAbvGr                  -0.135907
    BsmtFinType_Rec               -0.139119
    BsmtFinType_None              -0.145274
    BsmtFinType_BLQ               -0.146839
    Exterior_Wd_Sdng              -0.154201
    Exterior_MetalSd              -0.168906
    Exterior_PreCast                    NaN
    Name: SalePrice, dtype: float64



```python
#df['OverallQualSq'] = df['OverallQual']**2
#df['TotalFloorSpaceSq']= df['TotalFloorSpace'] ** 2
#df['GrLivAreaSq']=df['GrLivArea'] ** 2
#df['GarageCarsSq'] =df['GarageCars'] ** 2
#df['GarageAreaSq'] = df['GarageArea'] ** 2
#correlations= df.corr()['SalePrice'].sort_values(ascending=False )
#print(correlations)
```


```python

```


```python
fig = plt.figure(figsize= (10,10))

values_to_consider = ['SalePrice', 'OverallQual', 'TotalFloorSpace', 'GrLivArea','GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath','TotRmsAbvGrd', 'GarageYrBlt', 'YearRemodAdd', 'YearBuilt']
#sns.heatmap(df[['SalePrice', 'GarageArea', 'GarageCars']].corr(), annot=True)
sns.heatmap(df[values_to_consider].corr(), annot=True)


```

    /Users/rosscheung/anaconda/lib/python3.4/site-packages/matplotlib/collections.py:590: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if self._edgecolors == str('face'):





    <matplotlib.axes._subplots.AxesSubplot at 0x133baff60>




![png](House%20Price%20Prediction_files/House%20Price%20Prediction_84_2.png)


## Multicollinearity

It was found that dropping one of garagecars and garagearea greatly improves the prediction.  MMulticollinearity is s genuine concern! The same is true of "TotRmsAbvGrd" and "GrLivArea".


```python
df.drop('GarageArea',axis=1, inplace=True)

df.drop('TotRmsAbvGrd', axis=1,inplace=True)

#df.drop('GrLivArea', axis=1,inplace=True)

#What about garag year and

```


```python

```

## Let's fit the log(1+x) of salesprice

Sales price is initially skewed, but the log should fix this and make it gaussian.  This should be okay as long as we remember to do the inverse of this to our final modeled sales prices.

See: https://docs.scipy.org/doc/numpy/reference/generated/numpy.log1p.html


```python
df['SalePrice'] = np.log1p(df['SalePrice'])


```


```python
#hist(df['SalePrice'], 100)
plt.hist(df['SalePrice'].dropna(), 50)
```




    (array([   2.,    2.,    1.,    0.,    0.,    0.,    2.,    3.,    4.,
               3.,    5.,    1.,    5.,   21.,   22.,   23.,   18.,   29.,
              58.,   56.,   65.,  100.,  122.,   93.,   90.,   82.,  108.,
              91.,   64.,   55.,   58.,   51.,   46.,   42.,   23.,   29.,
              22.,   13.,   13.,   13.,    7.,    5.,    4.,    1.,    2.,
               2.,    2.,    0.,    0.,    2.]),
     array([ 10.46027076,  10.52175483,  10.5832389 ,  10.64472298,
             10.70620705,  10.76769112,  10.82917519,  10.89065926,
             10.95214334,  11.01362741,  11.07511148,  11.13659555,
             11.19807962,  11.25956369,  11.32104777,  11.38253184,
             11.44401591,  11.50549998,  11.56698405,  11.62846813,
             11.6899522 ,  11.75143627,  11.81292034,  11.87440441,
             11.93588849,  11.99737256,  12.05885663,  12.1203407 ,
             12.18182477,  12.24330884,  12.30479292,  12.36627699,
             12.42776106,  12.48924513,  12.5507292 ,  12.61221328,
             12.67369735,  12.73518142,  12.79666549,  12.85814956,
             12.91963363,  12.98111771,  13.04260178,  13.10408585,
             13.16556992,  13.22705399,  13.28853807,  13.35002214,
             13.41150621,  13.47299028,  13.53447435]),
     <a list of 50 Patch objects>)




![png](House%20Price%20Prediction_files/House%20Price%20Prediction_90_1.png)



```python
plt.scatter(df['SalePrice'], df['SizeDifferenceFromNeighbors'])


```




    <matplotlib.collections.PathCollection at 0x140935978>



    /Users/rosscheung/anaconda/lib/python3.4/site-packages/matplotlib/collections.py:590: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if self._edgecolors == str('face'):



![png](House%20Price%20Prediction_files/House%20Price%20Prediction_91_2.png)



```python
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(221)
df['TotalFloorSpace'].hist(bins=20)
ax.set_title('Total Floor Space')

ax = fig.add_subplot(222)
(df['GrLivArea']).hist(bins=20)
ax.set_title('Above grade (ground) living area square feet')

```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    /Users/rosscheung/anaconda/lib/python3.4/site-packages/pandas/indexes/base.py in get_loc(self, key, method, tolerance)
       2133             try:
    -> 2134                 return self._engine.get_loc(key)
       2135             except KeyError:


    pandas/index.pyx in pandas.index.IndexEngine.get_loc (pandas/index.c:4433)()


    pandas/index.pyx in pandas.index.IndexEngine.get_loc (pandas/index.c:4279)()


    pandas/src/hashtable_class_helper.pxi in pandas.hashtable.PyObjectHashTable.get_item (pandas/hashtable.c:13742)()


    pandas/src/hashtable_class_helper.pxi in pandas.hashtable.PyObjectHashTable.get_item (pandas/hashtable.c:13696)()


    KeyError: 'GrLivArea'


    During handling of the above exception, another exception occurred:


    KeyError                                  Traceback (most recent call last)

    <ipython-input-1142-a97cfd45e2c4> in <module>()
          6
          7 ax = fig.add_subplot(222)
    ----> 8 (df['GrLivArea']).hist(bins=20)
          9 ax.set_title('Above grade (ground) living area square feet')


    /Users/rosscheung/anaconda/lib/python3.4/site-packages/pandas/core/frame.py in __getitem__(self, key)
       2057             return self._getitem_multilevel(key)
       2058         else:
    -> 2059             return self._getitem_column(key)
       2060
       2061     def _getitem_column(self, key):


    /Users/rosscheung/anaconda/lib/python3.4/site-packages/pandas/core/frame.py in _getitem_column(self, key)
       2064         # get column
       2065         if self.columns.is_unique:
    -> 2066             return self._get_item_cache(key)
       2067
       2068         # duplicate columns & possible reduce dimensionality


    /Users/rosscheung/anaconda/lib/python3.4/site-packages/pandas/core/generic.py in _get_item_cache(self, item)
       1384         res = cache.get(item)
       1385         if res is None:
    -> 1386             values = self._data.get(item)
       1387             res = self._box_item_values(item, values)
       1388             cache[item] = res


    /Users/rosscheung/anaconda/lib/python3.4/site-packages/pandas/core/internals.py in get(self, item, fastpath)
       3541
       3542             if not isnull(item):
    -> 3543                 loc = self.items.get_loc(item)
       3544             else:
       3545                 indexer = np.arange(len(self.items))[isnull(self.items)]


    /Users/rosscheung/anaconda/lib/python3.4/site-packages/pandas/indexes/base.py in get_loc(self, key, method, tolerance)
       2134                 return self._engine.get_loc(key)
       2135             except KeyError:
    -> 2136                 return self._engine.get_loc(self._maybe_cast_indexer(key))
       2137
       2138         indexer = self.get_indexer([key], method=method, tolerance=tolerance)


    pandas/index.pyx in pandas.index.IndexEngine.get_loc (pandas/index.c:4433)()


    pandas/index.pyx in pandas.index.IndexEngine.get_loc (pandas/index.c:4279)()


    pandas/src/hashtable_class_helper.pxi in pandas.hashtable.PyObjectHashTable.get_item (pandas/hashtable.c:13742)()


    pandas/src/hashtable_class_helper.pxi in pandas.hashtable.PyObjectHashTable.get_item (pandas/hashtable.c:13696)()


    KeyError: 'GrLivArea'



![png](House%20Price%20Prediction_files/House%20Price%20Prediction_92_1.png)



```python
#Observation: The various "square feet" variables are also skewed.  Let's do the same for them
#df['TotalFloorSpace'] = np.log1p(df['TotalFloorSpace'])
#df['GrLivArea'] = np.log1p(df['GrLivArea'])
#df['TotalBsmtSF'] = np.log1p(df['TotalBsmtSF'])
#df['2ndFlrSF'] = np.log1p(df['2ndFlrSF'])
#df['1stFlrSF'] = np.log1p(df['1stFlrSF'])
#df['BsmtFinSF1'] = np.log1p(df['BsmtFinSF1'])
#df['BsmtFinSF2'] = np.log1p(df['BsmtFinSF2'])

#plt.hist(.dropna(), 100)

df['LotArea'] = np.log1p(df['LotArea'])
#sns.distplot(df['LotArea'])

```


```python
df.drop(['Neighborhood'], axis=1, inplace=True)
```


# Creating Dummy Variables

The dataset should be cleaned enough that we can create dummy variables.  Be warned, as you can see, the list of variables grows by a lot.


```python
df_dummies = pd.get_dummies(df, drop_first=True)

df_dummies.info()


```


```python
col_names = list(df_dummies.columns.values)

print(col_names)
#col_names
```


```python
#neighborhood_corr = df[['SalePrice', 'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr']]
```

## Let's split into dependent and independent variables


```python
dependent_vars = df_dummies['SalePrice']
df_houseprice = df_dummies[['Id', 'SalePrice']][1460:]


#not_sure = df_dummies[['MiscVal', 'Utilities_NoSeWa']]


#independent_vars = df_dummies[['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'CentralAir', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GarageYrBlt', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MSSubClass', 'MasVnrArea', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'PoolArea', 'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'YrSold', 'Alley_None', 'Alley_Pave', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'BsmtCond_Gd', 'BsmtCond_None', 'BsmtCond_Po', 'BsmtCond_TA', 'BsmtExposure_Gd', 'BsmtExposure_Mn', 'BsmtExposure_No', 'BsmtExposure_None', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_None', 'BsmtFinType1_Rec', 'BsmtFinType1_Unf', 'BsmtFinType2_BLQ', 'BsmtFinType2_GLQ', 'BsmtFinType2_LwQ', 'BsmtFinType2_None', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_None', 'BsmtQual_TA', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosA', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNe', 'Condition1_RRNn', 'Condition2_Feedr', 'Condition2_Norm', 'Condition2_PosA', 'Condition2_PosN', 'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_Mix', 'Electrical_SBrkr', 'ExterCond_Fa', 'ExterCond_Gd', 'ExterCond_Po', 'ExterCond_TA', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA', 'Exterior1st_AsphShn', 'Exterior1st_BrkComm', 'Exterior1st_BrkFace', 'Exterior1st_CBlock', 'Exterior1st_CemntBd', 'Exterior1st_HdBoard', 'Exterior1st_ImStucc', 'Exterior1st_MetalSd', 'Exterior1st_Plywood', 'Exterior1st_Stone', 'Exterior1st_Stucco', 'Exterior1st_VinylSd', 'Exterior1st_Wd Sdng', 'Exterior1st_WdShing', 'Exterior2nd_AsphShn', 'Exterior2nd_Brk Cmn', 'Exterior2nd_BrkFace', 'Exterior2nd_CBlock', 'Exterior2nd_CmentBd', 'Exterior2nd_HdBoard', 'Exterior2nd_ImStucc', 'Exterior2nd_MetalSd', 'Exterior2nd_Other', 'Exterior2nd_Plywood', 'Exterior2nd_Stone', 'Exterior2nd_Stucco', 'Exterior2nd_VinylSd', 'Exterior2nd_Wd Sdng', 'Exterior2nd_Wd Shng', 'Fence_GdWo', 'Fence_MnPrv', 'Fence_MnWw', 'Fence_None', 'FireplaceQu_Fa', 'FireplaceQu_Gd', 'FireplaceQu_None', 'FireplaceQu_Po', 'FireplaceQu_TA', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Sev', 'Functional_Typ', 'GarageCond_Fa', 'GarageCond_Gd', 'GarageCond_None', 'GarageCond_Po', 'GarageCond_TA', 'GarageFinish_None', 'GarageFinish_RFn', 'GarageFinish_Unf', 'GarageQual_Fa', 'GarageQual_Gd', 'GarageQual_None', 'GarageQual_Po', 'GarageQual_TA', 'GarageType_Attchd', 'GarageType_Basment', 'GarageType_BuiltIn', 'GarageType_CarPort', 'GarageType_Detchd', 'GarageType_None', 'Heating_GasA', 'Heating_GasW', 'Heating_Grav', 'Heating_OthW', 'Heating_Wall', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_Po', 'HeatingQC_TA', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl', 'LandSlope_Mod', 'LandSlope_Sev', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3', 'LotConfig_Inside', 'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'MSZoning_Unknown', 'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'MiscFeature_Othr', 'MiscFeature_Shed', 'MiscFeature_TenC', 'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards', 'Neighborhood_Gilbert', 'Neighborhood_IDOTRR', 'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NAmes', 'Neighborhood_NPkVill', 'Neighborhood_NWAmes', 'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'Neighborhood_OldTown', 'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW', 'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Neighborhood_Timber', 'Neighborhood_Veenker', 'PavedDrive_P', 'PavedDrive_Y', 'PoolQC_Fa', 'PoolQC_Gd', 'PoolQC_No Pool', 'RoofMatl_CompShg', 'RoofMatl_Membran', 'RoofMatl_Metal', 'RoofMatl_Roll', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'RoofStyle_Gable', 'RoofStyle_Gambrel', 'RoofStyle_Hip', 'RoofStyle_Mansard', 'RoofStyle_Shed', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal', 'SaleCondition_Partial', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'Street_Pave', 'Utilities_NoSeWa']]
vars_to_fit = col_names
vars_to_fit.remove('SalePrice')
vars_to_fit.remove('Id')
#vars_to_fit.remove('MiscVal')
#vars_to_fit.remove('MoSold')
vars_to_fit.remove('Utilities_NoSeWa')

independent_vars = df_dummies[vars_to_fit]
x_unknown = independent_vars[1460:]


x = independent_vars[0:1460]
y = dependent_vars[0:1460]


```


```python

```

### Now, after data cleaning but before splitting into test/train sets, let's remove those two outliers



```python
#x[x['GrLivArea']>5000][['GrLivArea', 'BsmtFinSF1', 'TotalBsmtSF', 'GarageArea']]
print(x[x['GrLivArea']>4500][['GrLivArea', 'BsmtFinSF1', 'TotalBsmtSF']])
outlier_index = x[x['GrLivArea']>4500][['GrLivArea', 'BsmtFinSF1', 'TotalBsmtSF']].index

x.drop(outlier_index, axis=0, inplace=True)
y.drop(outlier_index,inplace=True)

```


```python
x.info()
```


```python
x.reset_index(inplace=True)
x.drop('index',axis=1, inplace=True)
#y.reset_index(inplace=True)
#y.drop('index',axis=1, inplace=True)

```


```python

```


```python
y.shape
```

## Split training data into train/test datasets, so that we can diagnose how good each algorithm is


```python


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state = 0)
```

## Now let's perform our fits


```python
#parameters = { 'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0]} # could in theory test more but whatever
#clf = GridSearchCV(linear_model.Ridge(), parameters)
#clf.fit(x_train, y_train)
#clf.best_params_

```


```python
#ridgereg = linear_model.Ridge(alpha = 1.0) #alpha is learning rate,  Can change (to 0.01, etc)
ridgereg = linear_model.Ridge(alpha = 0.05) #alpha is learning rate,  Can change (to 0.01, etc)

ridgereg.fit(x_train, y_train)
y_pred_ridge = ridgereg.predict(x_test)
ridge_corr = ridgereg.score(x_test, y_test)
ridge_error = mean_squared_error(y_test, y_pred_ridge)**0.5

ridge_error, ridge_corr
#(0.11030102597135479, 0.92821064496608097)

```


```python
#lassoreg = linear_model.Lasso(alpha = .001)
lassoreg = linear_model.Lasso(alpha = .01)

lassoreg.fit(x_train, y_train)
y_pred_lasso = lassoreg.predict(x_test)
lasso_corr = lassoreg.score(x_test, y_test)
lasso_error = mean_squared_error(y_test, y_pred_lasso)**0.5

lasso_error, lasso_corr
```


```python
#parameters = {'learning_rate': [0.02, 0.03, 0.04, 0.05, 0.06], 'n_estimators': [ 1800, 1900, 2000, 2100,  2200]} # could in theory test more but whatever
#clf = GridSearchCV(GradientBoostingRegressor(), parameters)
#clf.fit(x_train, y_train)
#clf.best_params_

```


```python
# Do a gridsearch in sklearn
#
# Lasso does feature selection, then do gradient boosting?

# results of gridsearch
# {'learning_rate': 0.03, 'n_estimators': 2000}
#{'learning_rate': 0.05, 'n_estimators': 1900}
#'learning_rate': [0.02, 0.03, 0.04, 0.05, 0.06], 'n_estimators': [1700, 1800, 1900, 2000 ] {'learning_rate': 0.03, 'n_estimators': 1800}

#GradientBoosting = GradientBoostingRegressor(learning_rate = 0.02, n_estimators=3000, loss="huber")
#GradientBoosting = GradientBoosting.fit(x_train, y_train)
#y_pred_gradientboosting = GradientBoosting.predict(x_test)
#gradient_corr = GradientBoosting.score(x_test, y_test)

#gradient_error = mean_squared_error(y_test, y_pred_gradientboosting)**0.5


#gradient_error, gradient_corr
```


```python
#parameters = { 'n_estimators': [1900, 2000, 2100, 2200, 2300]} # could in theory test more but whatever
#clf = GridSearchCV(RandomForestRegressor(), parameters)
#clf.fit(x_train, y_train)
#clf.best_params_

```


```python


#RandomForest = RandomForestRegressor(n_estimators=1900)
#RandomForest = RandomForest.fit(x_train, y_train)
#y_pred_randomforest = RandomForest.predict(x_test)
#randomforest_corr = RandomForest.score(x_test, y_test)

#randomforest_error = mean_squared_error(y_test, y_pred_randomforest)**0.5


#randomforest_error, randomforest_corr

```


```python
#parameters = { 'learning_rate': [0.1, 0.5, 0.7, 1.0],'n_estimators': [1900, 2000, 2100]} # could in theory test more but whatever
#clf = GridSearchCV(AdaBoostRegressor(), parameters)
#clf.fit(x_train, y_train)
#clf.best_params_


#AdaBoost= AdaBoostRegressor(learning_rate= 1.0, n_estimators=2000)
#AdaBoost = AdaBoost.fit(x_train, y_train)
#y_pred_adaboost = AdaBoost.predict(x_test)
#adaboost_corr = AdaBoost.score(x_test, y_test)
#adaboost_error = mean_squared_error(y_test, y_pred_adaboost)**0.5
#adaboost_error, adaboost_corr

```


```python
#parameters = { 'learning_rate': [0.001, 0.01, 0.02, 0.1, 0.5, 0.7, 1.0],'n_estimators': [1000, 3000, 7200, 9000], } # could in theory test more but whatever
#clf = GridSearchCV(xgb.XGBRegressor(), parameters)
#clf.fit(x_train, y_train)
#clf.best_params_
#parameters = { 'learning_rate': [0.03],'n_estimators': [2500,2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000], } # could in theory test more but whatever
#clf = GridSearchCV(xgb.XGBRegressor(), parameters)
#clf.fit(x_train, y_train)
#clf.best_params_

```


```python
#xgboost = xgb.XGBRegressor(learning_rate = 0.03, n_estimators=1275, max_depth=3)
xgboost = xgb.XGBRegressor(learning_rate = 0.03, n_estimators=2600, max_depth=3)
xgboost = xgboost.fit(x_train, y_train)
y_pred_xgboost = xgboost.predict(x_test)
xgboost_corr = xgboost.score(x_test, y_test)

xgboost_error = mean_squared_error(y_test, y_pred_xgboost)**0.5


xgboost_error, xgboost_corr

#(0.11091205185460548, 0.92741307029539688)
#(0.11286956997136902, 0.92482824524109142)

#(0.11129650831565363, 0.92690897934993766)
#(0.10996132516749133, 0.92865215216847619) 0.03, 4000
#(0.10991824002441893, 0.92870805236778797) 0.03, 3000
#(0.10961827680696469, 0.92909662814121641) 0.03, 2000
#(0.10916191236725561, 0.92968577116229045) 0.03, 1500
#(0.10911116020722886, 0.92975113772376539) 0.03, 1250


```


```python
#### enet = ElasticNet(alpha = 1.0, l1_ratio=0.1)
#enet = enet.fit(x_train, y_train)
#y_pred_enet = enet.predict(x_test)
#enet_corr = enet.score(x_test, y_test)

#enet_error = mean_squared_error(y_test, y_pred_xgboost) **0.5

#enet_error, enet_corr
```


```python
#parameters = {'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
#clf = GridSearchCV(ElasticNet(), parameters)
#clf.fit(x_train, y_train)
#clf.best_params_

```


```python

#y_ensemble_pred = (y_pred_gradientboosting/gradient_error**2 + y_pred_randomforest/randomforest_error**2 + y_pred_xgboost/xgboost_error**2+ y_pred_ridge/ridge_error**2+y_pred_lasso/lasso_error**2 )
#y_ensemble_pred = y_ensemble_pred / (1/randomforest_error**2 + 1/gradient_error**2 + 1/ridge_error**2 + 1/lasso_error**2  + 1/xgboost_error**2 )
y_ensemble_pred = ( y_pred_xgboost/xgboost_error**2+ y_pred_ridge/ridge_error**2+y_pred_lasso/lasso_error**2 )
y_ensemble_pred = y_ensemble_pred / ( 1/ridge_error**2 + 1/lasso_error**2  + 1/xgboost_error**2 )


ensemble_error = mean_squared_error(y_test, y_ensemble_pred)**0.5

ensemble_error
```


```python
# Current highest scoring (read: lowest) ensemble error
#0.099778052606703235

#0.099586287880297747
# With Lasso and Ridge scores set quite low


```

### Unsurprisingly the ensemble methods vastly outperform the linear ones

Now, output to a CSV, and remember to undo the log(x+1) done previously


```python
#GradientBoosting = GradientBoosting.fit(x, y) # Fit again since we only used 80% of the data last time
#y_gradientboosting = np.expm1(GradientBoosting.predict(x_unknown))
#df_houseprice['SalePrice'] =  y_gradientboosting
#df_houseprice.to_csv('output_gradient_boosting.csv',index=False)

#RandomForest = RandomForest.fit(x, y) # Fit again since we only used 80% of the data last time
#y_randomforest = np.expm1(RandomForest.predict(x_unknown))
#df_houseprice['SalePrice'] =  y_randomforest
#df_houseprice.to_csv('output_random_forest.csv',index=False)


xgboost = xgboost.fit(x, y)
y_xgboost = np.expm1(xgboost.predict(x_unknown))
df_houseprice['SalePrice'] =  y_xgboost
df_houseprice.to_csv('output_XGboost.csv',index=False)

#AdaBoost = AdaBoost.fit(x,y)
#y_adaboost = np.expm1(AdaBoost.predict(x_unknown))
#df_houseprice['SalePrice'] =  y_adaboost
#df_houseprice.to_csv('output_adaboost.csv',index=False)


ridgereg = ridgereg.fit(x,y)
y_ridgereg = np.expm1(ridgereg.predict(x_unknown))
#df_houseprice['SalePrice'] =  y_adaboost
#df_houseprice.to_csv('output_adaboost.csv',index=False)

lassoreg = lassoreg.fit(x,y)
y_lassoreg = np.expm1(lassoreg.predict(x_unknown))

```


```python
#temp = GradientBoosting.predict(x_unknown)
#df_houseprice['SalePrice'].plot('hist', 100)
#temp
```


```python
# To do: revisit "condition"
# Take the categorical variables which should be ordered, and make them ordered/numerical
# Calculate log of sales price, etc
# Neighborhood - do feature engineering on it
# Will dropping some highly correlated variables help?  
# Raising more correlated features by power of 2 or 3
#
# Elastic Net? Extra Trees?

# Order those non-equal categorical variables!
# Bed/Bath ratio
df_houseprice.head(20)
```


```python
# log average
#ensemble = np.exp((np.log(y_gradientboosting) + np.log(y_randomforest) + np.log(y_xgboost)+np.log(y_adaboost) + np.log(y_ridgereg)+np.log(y_lassoreg))/6)

#More standard mean

#ensemble = ((y_gradientboosting + y_randomforest + y_xgboost+y_adaboost + y_ridgereg+y_lassoreg)/6)

#ensemble = (y_gradientboosting/gradient_error + y_randomforest/randomforest_error + y_xgboost/xgboost_error+y_adaboost/adaboost_error + y_ridgereg/ridge_error+y_lassoreg/lasso_error)
#ensemble = ensemble / (1/randomforest_error + 1/gradient_error + 1/ridge_error + 1/lasso_error + 1/adaboost_error+ 1/xgboost_error)


#ensemble = (y_gradientboosting/gradient_error**2 + y_randomforest/randomforest_error**2 + y_xgboost/xgboost_error**2+ y_ridgereg/ridge_error**2+y_lassoreg/lasso_error**2)
#ensemble = ensemble / (1/randomforest_error**2 + 1/gradient_error**2 + 1/ridge_error**2 + 1/lasso_error**2  + 1/xgboost_error**2)

ensemble = ( y_xgboost/xgboost_error**2+ y_ridgereg/ridge_error**2+y_lassoreg/lasso_error**2)
ensemble = ensemble / ( 1/ridge_error**2 + 1/lasso_error**2  + 1/xgboost_error**2)


```


```python
df_houseprice['SalePrice'] =  ensemble
df_houseprice.to_csv('output_ensemble.csv',index=False)


xstuff = np.linspace(1.,len(y_xgboost),len(y_xgboost))
plt.plot(xstuff, ensemble, 'b-', xstuff, y_xgboost, 'r-')
```

## Let's try model stacking

Some resources consulted for this section:

https://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/

https://mlwave.com/kaggle-ensembling-guide/

Elements of Statistical Learning (http://web.stanford.edu/~hastie/ElemStatLearn/)


```python
kfold = KFold(n_splits=5, shuffle = True)

```


```python
#allFolds = kfold.split(x,y)
```


```python

```


```python
allBasePredictions = np.zeros((x.shape[0],3))  # One column for every base model to train
allBasePredictions
```

There's a weird dataframe to numpy conversion that's necessary to avoid an out of bounds error.  See below link:

https://stackoverflow.com/questions/30023927/sklearn-cross-validation-stratifiedshufflesplit-error-indices-are-out-of-bou


```python
x_numpy = x.values
for k, (index_train, index_test) in enumerate(kfold.split(x,y)):
 #   index_train.reshape(1, len(index_train))
    first = x.loc[index_train].reset_index()
    second = x.loc[index_test].reset_index()#.reindex()
    lassoreg.fit(x.iloc[index_train], y[index_train])
    y_predicted = lassoreg.predict(x.iloc[index_test])
    allBasePredictions[index_test,0] = y_predicted

#x_numpy = x.values
#for k, (index_train, index_test) in enumerate(kfold.split(x,y)):
#    ridgereg.fit(x_numpy[index_train], y[index_train])
#    y_predicted = ridgereg.predict(x_numpy[index_test])
#    allBasePredictions[index_test,1] = y_predicted

#x_numpy = x.values
#for k, (index_train, index_test) in enumerate(kfold.split(x,y)):
#    xgboost.fit(x_numpy[index_train], y[index_train])
#    y_predicted = xgboost.predict(x_numpy[index_test])
#    allBasePredictions[index_test,2] = y_predicted

```


```python
first.isnull().any()
```


```python

```


```python
first = x.loc[index_train].reset_index()

first.drop(['index'], axis=1, inplace=True)
first
```


```python
x.loc[index_train].isnull()[1030:1040]
```


```python
x[1290:1300]
```


```python
index_train[1030:1040]
x.iloc[index_train[1030:1040]]
```


```python
first[1030:1040]
```


```python
z = y[index_train]
z
```


```python
lassoreg.fit(first.dropna, z)
```


```python
first.head()
```


```python
first[first['1stFlrSF'].isnull()]
```


```python
http://stackoverflow.com/questions/21320456/scikit-nan-or-infinity-error-message


```


```python
x[1034]
```


```python

```
