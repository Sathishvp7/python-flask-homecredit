from flask import Flask, request, jsonify, render_template
#importing Useful DataStructures
import pandas as pd
import numpy as np

#importing Misc Libraries
import os
import gc
import pickle
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

#sklearn
from sklearn.metrics import accuracy_score,precision_score,f1_score,roc_auc_score,recall_score

# Import all required package for this problem
import matplotlib.pyplot as plt
#import seaborn as sns
#from plotly.offline import iplot
#from matplotlib.pyplot import figure
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, LabelEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer as Imputer

#import plotly.offline as py
#import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
#import cufflinks as cf
#cf.go_offline()
#import lightgbm as lgb
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm



app = Flask(__name__)
model = pickle.load(open(r'F:\Softwares\Deployment-flask-master\model.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')
	
	



def reduce_mem_usage(data, verbose = False):
    # refer - https://medium.com/@aakashgoel12/avoid-memory-error-techniques-to-reduce-dataframe-memory-usage-fcf53b2318a2
    #refer: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
    '''
    This function is used to reduce the memory usage by converting the datatypes of a pandas
    DataFrame withing required limits.
    '''
    
    start_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('-'*100)
        print('Memory usage of dataframe: {:.2f} MB'.format(start_mem))
    
    for col in data.columns:
        col_type = data[col].dtype
        
        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

    end_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage after optimization: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        print('-'*100)

    return data



# Read all madatory files for this task
def read_mandatory_files(test_data='test_dataframe'):
    #df = pd.read_csv('application_train.csv')
    #df = reduce_mem_usage(df)
    #df_test = pd.read_csv('application_test.csv')
    #df_test = reduce_mem_usage(df_test)
    #print(type(test_data))
    df_test = test_data
    df_test = reduce_mem_usage(df_test)

    df_bureau = pd.read_csv('bureau.csv')
    df_bureau = reduce_mem_usage(df_bureau)
    df_bureau_bal = pd.read_csv('bureau_balance.csv')
    df_bureau_bal = reduce_mem_usage(df_bureau_bal)
    df_prev_app = pd.read_csv('previous_application.csv')
    df_prev_app = reduce_mem_usage(df_prev_app)
    df_pos = pd.read_csv('POS_CASH_balance.csv')
    df_pos = reduce_mem_usage(df_pos)
    df_credit_bal = pd.read_csv('credit_card_balance.csv')
    df_credit_bal = reduce_mem_usage(df_credit_bal)
    df_inst_pay = pd.read_csv('installments_payments.csv')
    df_inst_pay = reduce_mem_usage(df_inst_pay)
    
    return df_test, df_bureau,df_bureau_bal,df_prev_app,df_pos, df_credit_bal,df_inst_pay



# Data Cleaning and preprocessing for train and test
def data_cleaning(df='dataframe'):
    df['DAYS_BIRTH'] = round(df['DAYS_BIRTH'] *-1/ 365)

    # abs - convert all value in to postive
    df['DAYS_EMPLOYED']  = abs(df['DAYS_EMPLOYED'])
    df['DAYS_EMPLOYED'].head(2)

    # Replace the anomalous values(Errorness value) with nan
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({365243: np.nan})

    # Convert days in to years
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'] /365
    
    # Create an anomalous flag column
    df['Year_Empolyed_ANOM'] = df["DAYS_EMPLOYED"] == 365243

    #Invalid Gender code, we have limited entry so we removing
    df = df[df['CODE_GENDER'] != 'XNA']

    df.loc[df['OBS_30_CNT_SOCIAL_CIRCLE'] > 30, 'OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan
    df.loc[df['OBS_60_CNT_SOCIAL_CIRCLE'] > 30, 'OBS_60_CNT_SOCIAL_CIRCLE'] = np.nan


    
    return df



# replacing Null with XNS -  Unknow category
def replacing_missing_category(df = 'dataframe'):
    # replacing Null with XNS -  Unknow category
    categorical_columns_train = df.dtypes[df.dtypes == 'object'].index.tolist()
    df[categorical_columns_train] = df[categorical_columns_train].fillna('XNA')
    
    # From EDA REGION_RATING_CLIENT and REGION_RATING_CLIENT_W_CITY have discret value, 
    #so we changing this column data type from Int to Object
    
    df['REGION_RATING_CLIENT'] = df['REGION_RATING_CLIENT'].astype('object')
    df['REGION_RATING_CLIENT_W_CITY'] = df['REGION_RATING_CLIENT_W_CITY'].astype('object')

    
    return df



## Converting Category in to Numerical representation using Onehot encoder and label encoder
'''
OneHotEncoder -  handle_unknown='ignore'
    unknown category is encountered during
    transform, the resulting one-hot encoded columns for this feature
    will be all zeros.
'''
def category_to_numeric(df_test = 'dataframe2'):
    #print('Shape of application test before Encoding ', df_test.shape)
    cat_df = df_test.dtypes[df_test.dtypes == 'object'].index.tolist()
    #enc = OneHotEncoder(handle_unknown='ignore')
    for i in tqdm(cat_df):
        #enc = pickle.load('Encoder//'+str(i)+'_onehot.pkl') 
        with open(('Encoder//'+str(i)+'_onehot.pkl') , 'rb') as f:
          enc = pickle.load(f)
        enc_df_test = pd.DataFrame(enc.transform(df_test[[i]]).toarray().astype('int'),columns=enc.get_feature_names([i]))
        df_test = df_test.drop(i,axis=1).join(enc_df_test)

    #print('Shape of application test after Encoding ', df_test.shape)  
    
    return df_test

# replacing ,missing value with Median
def replacing_missing_numeric(df = 'dataframe', df_test='dataframe2'):

    # Test missing values
    #missing_value(df_test,df_name='TEST',visualizse=False,head_count=10)
    
    ## Fill in missing values

    #Strategy = Median, variances is high so better to use Median
    imputer = Imputer(strategy = 'median')
    scaler = MinMaxScaler(feature_range = [0,1])
    train = df_test
    train_col = train.columns


    with open((r'Imputer_folder/_Imputer.pkl') , 'rb') as f:
      imputer = pickle.load(f)
      
    test = imputer.transform(df_test)
    with open((r'Scalar/_ScalarImputer.pkl') , 'rb') as f:
      scaler = pickle.load(f)    
    test = scaler.transform(test)
    #print('Testing data shape: ', test.shape)
    
    new_df_test = pd.DataFrame(test,columns=train_col)
    new_df_test['SK_ID_CURR'] = df_test['SK_ID_CURR'].values
    
    # Test missing values
    #missing_value(new_df_test,df_name='TEST',visualizse=False,head_count=5)

    #print(' Observation : \n 1.Now there is no missing value in Train and test')
    
    
    return  new_df_test



def create_custom_features_main_table( df_test='dataframe2'):


    ################ Application_ test.csv ###########

    df_test['INCOME_GT_CREDIT_FLAG'] = df_test['AMT_INCOME_TOTAL'] > df_test['AMT_CREDIT']
    df_test['DIR'] = df_test['AMT_CREDIT']/(df_test['AMT_INCOME_TOTAL']+ 1)
    df_test['AIR'] = df_test['AMT_ANNUITY']/(df_test['AMT_INCOME_TOTAL'] +1)
    df_test['ACR'] = df_test['AMT_CREDIT']/(df_test['AMT_CREDIT']+1)
    df_test['DAR'] = df_test['DAYS_EMPLOYED']/df_test['DAYS_BIRTH']
    
    return df_test




# when merging two table, chance o having same column name in both table, to avoid we settiing new feature name
# Note , pd.Dummies we have prefix attribute, but for numerical field better to use this function
def create_unique_col(table= 'Bureau',data='df', ID ='SK_ID_CURR' ):
    '''
    table - Dataframe name
    data  = Dataframe
    ID    = Foreign key
    return Column name wit prefix table name
    '''
    
    unique_col_bureau = [] 
    for i in data.columns:
        if i != ID:
            col_name = table+str('_')+str(i)
            unique_col_bureau.append(col_name)
        else:
            unique_col_bureau.append(i)
    #print('New Column names  - \n'+str(unique_col_bureau))   
    return unique_col_bureau  



def create_custom_bureau_feature(df_bureau='bureau_table'):
    # Create a new column , using existing information from Bureau
    # Number of past loans per customer
    past_loan = df_bureau.groupby(by = ['SK_ID_CURR'])['SK_ID_BUREAU'].count().reset_index().rename(columns = {'SK_ID_BUREAU': 'LOAN_COUNT_BUREAU'})
    #print('Past loan details',past_loan.shape )


    # Number of type of credit loan type per customer
    credit_type = df_bureau[['SK_ID_CURR', 'CREDIT_TYPE']].groupby('SK_ID_CURR')['CREDIT_TYPE'].nunique().reset_index().rename(columns={'CREDIT_TYPE': 'LOAN_TYPES_BUREAU'})
    #print('credit_type details',credit_type.shape)


    # total_loan amount still date
    sum_total_count = df_bureau[['SK_ID_CURR', 'AMT_CREDIT_SUM']].groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM': 'TOTAL_LOAN_AMT_SUM'})
    #print('sum_total_count details',sum_total_count.shape )


    # total_loan amount debt still date
    sum_total_count_debt = df_bureau[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_LOAN_AMT_SUM_DEBT'})
    #print('sum_total_count_debt details',sum_total_count_debt.shape)


    #merge sum_total_count and sum_total_count_debt
    debt_credit_df  =  sum_total_count.merge(sum_total_count_debt,on='SK_ID_CURR')
    debt_credit_df['debt_credit_ratio'] = debt_credit_df['TOTAL_LOAN_AMT_SUM_DEBT'] / (debt_credit_df['TOTAL_LOAN_AMT_SUM'] +1)
    #print('Merge of sum_total_count and sum_total_count_debt',debt_credit_df.shape)


    # Sum of AMT_CREDIT_SUM_OVERDUE
    Total_customer_overdue  = df_bureau[['SK_ID_CURR', 'AMT_CREDIT_SUM_OVERDUE']].groupby('SK_ID_CURR')['AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM_OVERDUE': 'SUM_AMT_CREDIT_SUM_OVERDUE'})
    #print('Sum of over due amount',Total_customer_overdue.shape)
    # sum AMT_CREDIT_SUM_DEBT
    Total_customer_debt =  df_bureau[['SK_ID_CURR','AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM_DEBT':'TOTAL_CUSTOMER_DEBT'})
    #print('Sum of total debt amount ',Total_customer_debt.shape)


    sum_overdue_credit_sum  = Total_customer_overdue.merge(Total_customer_debt,on='SK_ID_CURR')
    sum_overdue_credit_sum['overdue_debt_ratio'] = sum_overdue_credit_sum['SUM_AMT_CREDIT_SUM_OVERDUE'] / (sum_overdue_credit_sum['TOTAL_CUSTOMER_DEBT']+1)
    sum_overdue_credit_sum['overdue_debt_ratio'] = sum_overdue_credit_sum['overdue_debt_ratio'].fillna(0)
    sum_overdue_credit_sum['overdue_debt_ratio'] = sum_overdue_credit_sum.replace([np.inf,-np.inf],0)
    sum_overdue_credit_sum['overdue_debt_ratio'] = pd.to_numeric(sum_overdue_credit_sum['overdue_debt_ratio'],downcast='float')
    #print('Ratio of Overdue and credit debt amount',sum_overdue_credit_sum.shape)
    
    df_bureau = df_bureau.merge(past_loan,on='SK_ID_CURR',how='left')
    df_bureau =df_bureau.merge(credit_type,on='SK_ID_CURR',how='left')
    df_bureau =df_bureau.merge(debt_credit_df,on='SK_ID_CURR',how='left')
    df_bureau =df_bureau.merge(sum_overdue_credit_sum,on='SK_ID_CURR',how='left')
    
    return df_bureau





# Joining application_train and BUREAU.csv
def join_application_bureau(df ='table1', df_bureau = 'table2'):
    # Categorical feature - merging
    # Converting all categorical in to onehot encoding
    categorical_bureau = pd.get_dummies(df_bureau.select_dtypes('object'), prefix='Bureau')
    categorical_bureau['SK_ID_CURR'] = df_bureau['SK_ID_CURR']

    grp_bureau = categorical_bureau.groupby(by = ['SK_ID_CURR']).mean().reset_index()
    #print('Column_names_Categorical', grp_bureau.columns)
    
    # Merge train and bureau_categorical
    df_main = df.merge(grp_bureau, on='SK_ID_CURR',how='left')
    df_main.update(df_main[grp_bureau.columns].fillna(0))
    
    # Combining Numerical features

    Numerical_bureau_col = df_bureau.select_dtypes(include=[np.number]).columns
    Numerical_bureau = df_bureau[Numerical_bureau_col]


    grp_bureau_num = Numerical_bureau.drop(['SK_ID_BUREAU'], axis = 1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    #print('Column_names_Numeric', grp_bureau_num.columns)

    #There may be some column name for both application and bureau.csv, inorder to avoid we giving unique column name
    grp_bureau_num.columns = create_unique_col(table='Bureau',data=grp_bureau_num,ID='SK_ID_CURR')

    # Merge train and bureau_categorical
    df_main = df_main.merge(grp_bureau_num, on='SK_ID_CURR',how='left')
    df_main.update(df_main[grp_bureau_num.columns].fillna(0))
    
    return df_main






# Joining Bureau_Balance data to merge of Application and Bureau(df_main)

def data_cleaning_bureau_bal(df_bureau_bal= 'dataframe'):
    # C - Close , so we giving weight to 0, remaining incremental, thing is X - Unknow so benefit of doubt we giving middle value(4) 
    status_label_encoding = { 'C': 0, '0': 1, '1': 2, '2': 3, 'X': 4, '3': 5, '4': 6, '5': 7}
    df_bureau_bal['STATUS'] = df_bureau_bal['STATUS'].map(status_label_encoding)

    # Monthly Balance is in Negative , easy interpreatation we changing to postive
    df_bureau_bal['MONTHS_BALANCE'] = abs(df_bureau_bal['MONTHS_BALANCE'])

    # Creating new features 'Weightage_balance' = divide Status by Months_balance 
    df_bureau_bal['WEIGHT_status'] = df_bureau_bal['STATUS'] / (df_bureau_bal['MONTHS_BALANCE'] +1)
    
    return df_bureau_bal

def merge_application_BureauBal(df_main='dataframe1', df_bureau='dataframe2', df_bureau_bal='dataframe3'):
    df_bureau_bal = data_cleaning_bureau_bal(df_bureau_bal= df_bureau_bal)
    Bureau_merge_Bureau_bal =  df_bureau.merge(df_bureau_bal, on='SK_ID_BUREAU')
    Bureau_merge_Bureau_bal = Bureau_merge_Bureau_bal[['SK_ID_CURR', 'MONTHS_BALANCE','STATUS','WEIGHT_status']].groupby('SK_ID_CURR')['MONTHS_BALANCE','STATUS','WEIGHT_status'].sum().reset_index()
    Bureau_merge_Bureau_bal.columns = create_unique_col(table='Bureau_bal',data=Bureau_merge_Bureau_bal,ID='SK_ID_CURR')
    df_main = df_main.merge(Bureau_merge_Bureau_bal, on='SK_ID_CURR',how='left')
    df_main.update(df_main[['Bureau_bal_MONTHS_BALANCE','Bureau_bal_STATUS','Bureau_bal_WEIGHT_status']].fillna(0))
    #print('Shape of main table after merge Application, Bureau and Bureau_balance', df_main.shape)
    
    return df_main



def create_custom_prevapp_feature(previous_application='bureau_table'):
    # Create a new column , using existing information from Previous applications
    #https://www.kaggle.com/c/home-credit-default-risk/discussion/64598
    previous_application['AMT_INTEREST'] = previous_application['CNT_PAYMENT'] * previous_application[
                                            'AMT_ANNUITY'] - previous_application['AMT_CREDIT'] 
    previous_application['INTEREST_SHARE'] = previous_application['AMT_INTEREST'] / (previous_application[
                                                                                            'AMT_CREDIT'] + 0.00001)
    previous_application['INTEREST_RATE'] = 2 * 12 * previous_application['AMT_INTEREST'] / (previous_application[
                                        'AMT_CREDIT'] * (previous_application['CNT_PAYMENT'] + 1))
    

    previous_application['AMT_DECLINED'] = previous_application['AMT_APPLICATION'] - previous_application['AMT_CREDIT']

    previous_application['AMT_CREDIT_GOODS_RATIO'] = previous_application['AMT_CREDIT'] / (previous_application['AMT_GOODS_PRICE'] + 0.00001)
    previous_application['AMT_CREDIT_GOODS_DIFF'] = previous_application['AMT_CREDIT'] - previous_application['AMT_GOODS_PRICE']

    previous_application['ANNUITY'] = previous_application['AMT_CREDIT'] / (previous_application['CNT_PAYMENT'] + 0.00001)
    previous_application['ANNUITY_GOODS'] = previous_application['AMT_GOODS_PRICE'] / (previous_application['CNT_PAYMENT'] + 0.00001)
   
    #print('After creating custom feature ', previous_application.shape)
    return previous_application





def merge_application_prev_app(df_main = 'dataframe1',df_prev_app ='dataframe2'):
    Pre_app_count= df_prev_app[['SK_ID_CURR','SK_ID_PREV']].groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].count().reset_index().rename(columns={'SK_ID_PREV':'PREV_APP_COUNT'}).fillna(0)

    # Combining categorical features
    pre_app_categorical = pd.get_dummies(df_prev_app.select_dtypes('object'))
    pre_app_categorical['SK_ID_CURR'] = df_prev_app['SK_ID_CURR']

    grp_PrevApp = pre_app_categorical.groupby('SK_ID_CURR').mean().reset_index()
    grp_PrevApp.columns = create_unique_col(table='PREV_APP',data=grp_PrevApp,ID='SK_ID_CURR')

    # Combine final Previous_Application to df_main
    df_main = df_main.merge(grp_PrevApp,on='SK_ID_CURR',how='left')
    df_main.update(df_main[grp_PrevApp.columns].fillna(0))

    # Combining numerical features
    grp_PrevApp_numeric = df_prev_app.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    grp_PrevApp_numeric.columns = create_unique_col(table='PREV_APP',data=grp_PrevApp_numeric,ID='SK_ID_CURR')
    df_main = df_main.merge(grp_PrevApp_numeric, on =['SK_ID_CURR'], how = 'left')
    df_main.update(df_main[grp_PrevApp_numeric.columns].fillna(0))

    #print('Shape after merge Application, Bureau, Bureau_Balance and Previous_Application ', df_main.shape)
    return df_main
    
    



def create_custom_pos_feature(pos_cash='dataframe1'):
    
    #creating new features based on Domain Knowledge
    pos_cash['SK_DPD_RATIO'] = pos_cash['SK_DPD'] / (pos_cash['SK_DPD_DEF'] + 0.00001)

    pos_cash['TOTAL_TERM'] = pos_cash['CNT_INSTALMENT'] + pos_cash['CNT_INSTALMENT_FUTURE']

    #print('Shape of POS after feature engineering', pos_cash.shape)

    return pos_cash




def merge_application_pos(df_main = 'dataframe1',df_pos ='dataframe2'):
    #Pre_app_count= df_prev_app[['SK_ID_CURR','SK_ID_PREV']].groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].count().reset_index().rename(columns={'SK_ID_PREV':'PREV_APP_COUNT'}).fillna(0)

    POS_count= df_pos[['SK_ID_CURR','SK_ID_PREV']].groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].count().reset_index().rename(columns={'SK_ID_PREV':'POS_COUNT'}).fillna(0)

    # Combining categorical features
    POS_categorical = pd.get_dummies(df_pos.select_dtypes('object'))
    POS_categorical['SK_ID_CURR'] = df_pos['SK_ID_CURR']

    grp_POS = POS_categorical.groupby('SK_ID_CURR').mean().reset_index()
    grp_POS.columns = create_unique_col(table='POS',data=grp_POS,ID='SK_ID_CURR')

    # Combine final Previous_Application to df_main
    df_main = df_main.merge(grp_POS,on='SK_ID_CURR',how='left')
    df_main.update(df_main[grp_POS.columns].fillna(0))

    # Combining numerical features
    POS_numeric = df_pos.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    POS_numeric.columns = create_unique_col(table='POS',data=POS_numeric,ID='SK_ID_CURR')
    df_main = df_main.merge(POS_numeric, on =['SK_ID_CURR'], how = 'left')
    df_main.update(df_main[POS_numeric.columns].fillna(0))

    #print('Shape after merge Application, Bureau, Bureau_Balance , Previous_Application and POS ', df_main.shape)
    
    return df_main
    
    



def create_custom_Inspay_feature(installments_payments='bureau_table'):
    # Create a new column , using existing information from Install payment applications
    installments_payments['AMT_PAYMENT_DIFF'] = installments_payments['AMT_INSTALMENT'] - installments_payments['AMT_PAYMENT']
    installments_payments['AMT_PAYMENT_RATIO'] = installments_payments['AMT_PAYMENT'] / (installments_payments['AMT_INSTALMENT'] + 0.00001)
    installments_payments['DAYS_PAYMENT_RATIO'] = installments_payments['DAYS_INSTALMENT'] / (installments_payments['DAYS_ENTRY_PAYMENT'] + 0.00001)
    installments_payments['DAYS_PAYMENT_DIFF'] = installments_payments['DAYS_INSTALMENT'] - installments_payments['DAYS_ENTRY_PAYMENT']

    #print('Shape of InstallPayment after feature engineering', installments_payments.shape)

    return installments_payments
   





def merge_application_Inspay(df_main = 'dataframe1',df_inst_pay ='dataframe2'):
    INSPAY_numeric = df_inst_pay.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    INSPAY_numeric.columns = create_unique_col(table='INSTPAY',data=INSPAY_numeric,ID='SK_ID_CURR')
    df_main = df_main.merge(INSPAY_numeric, on =['SK_ID_CURR'], how = 'left')
    df_main.update(df_main[INSPAY_numeric.columns].fillna(0))

    #print('Shape after merge Application, Bureau, Bureau_Balance , Previous_Application, POS and InstantPay ', df_main.shape)
    return df_main



def create_custom_Credit_bal_feature(cc_balance='CCB_table'):

     #Creating new features
        cc_balance['AMT_DRAWING_SUM'] = cc_balance['AMT_DRAWINGS_ATM_CURRENT'] + cc_balance['AMT_DRAWINGS_CURRENT'] + cc_balance[
                                    'AMT_DRAWINGS_OTHER_CURRENT'] + cc_balance['AMT_DRAWINGS_POS_CURRENT']
        cc_balance['BALANCE_LIMIT_RATIO'] = cc_balance['AMT_BALANCE'] / (cc_balance['AMT_CREDIT_LIMIT_ACTUAL'] + 0.00001)
        cc_balance['CNT_DRAWING_SUM'] = cc_balance['CNT_DRAWINGS_ATM_CURRENT'] + cc_balance['CNT_DRAWINGS_CURRENT'] + cc_balance[
                                            'CNT_DRAWINGS_OTHER_CURRENT'] + cc_balance['CNT_DRAWINGS_POS_CURRENT'] + cc_balance['CNT_INSTALMENT_MATURE_CUM']
        cc_balance['MIN_PAYMENT_RATIO'] = cc_balance['AMT_PAYMENT_CURRENT'] / (cc_balance['AMT_INST_MIN_REGULARITY'] + 0.0001)
        cc_balance['PAYMENT_MIN_DIFF'] = cc_balance['AMT_PAYMENT_CURRENT'] - cc_balance['AMT_INST_MIN_REGULARITY']
        cc_balance['MIN_PAYMENT_TOTAL_RATIO'] = cc_balance['AMT_PAYMENT_TOTAL_CURRENT'] / (cc_balance['AMT_INST_MIN_REGULARITY'] +0.00001)
        cc_balance['PAYMENT_MIN_DIFF'] = cc_balance['AMT_PAYMENT_TOTAL_CURRENT'] - cc_balance['AMT_INST_MIN_REGULARITY']
        cc_balance['AMT_INTEREST_RECEIVABLE'] = cc_balance['AMT_TOTAL_RECEIVABLE'] - cc_balance['AMT_RECEIVABLE_PRINCIPAL']
        cc_balance['SK_DPD_RATIO'] = cc_balance['SK_DPD'] / (cc_balance['SK_DPD_DEF'] + 0.00001)

        #print('Shape of Credit Card balance after feature engineering', cc_balance.shape)

        return cc_balance



def merge_application_credit_bal(df_main = 'dataframe1',df_credit_bal ='dataframe2'):
    # Combining categorical features
    CREBAL_categorical = pd.get_dummies(df_credit_bal.select_dtypes('object'))
    CREBAL_categorical['SK_ID_CURR'] = df_credit_bal['SK_ID_CURR']

    grp_CREDBAL = CREBAL_categorical.groupby('SK_ID_CURR').mean().reset_index()
    grp_CREDBAL.columns = create_unique_col(table='CREDITBAL',data=grp_CREDBAL,ID='SK_ID_CURR')

    # Combine final Previous_Application to df_main
    df_main = df_main.merge(grp_CREDBAL,on='SK_ID_CURR',how='left')
    df_main.update(df_main[grp_CREDBAL.columns].fillna(0))

    # Combining numerical features
    CREDBAL_numeric = df_credit_bal.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()
    CREDBAL_numeric.columns = create_unique_col(table='CREDITBAL',data=CREDBAL_numeric,ID='SK_ID_CURR')
    df_main = df_main.merge(CREDBAL_numeric, on =['SK_ID_CURR'], how = 'left')
    df_main.update(df_main[CREDBAL_numeric.columns].fillna(0))

    #print('Shape after merge Application, Bureau, Bureau_Balance , Previous_Application, POS, INSTALLMENT PAYMENT and CREDIt BAL ', df_main.shape)
    
    return df_main




def final_function_1(input_data):
    '''
    Preprocessing the test dataframe -  input_data: DataFrame
        The test datapoint, whose Target is to be predicted
    '''
    df_test, df_bureau,df_bureau_bal,df_prev_app,df_pos, df_credit_bal,df_inst_pay = read_mandatory_files(test_data=input_data)
    print('Shape of Test1', df_test.shape)
    # Data Cleaning
    df_test =  data_cleaning(df = df_test)
    print('Shape of Test2', df_test.shape)
    
    # Handling missing value-  Category
    df_test =  replacing_missing_category(df = df_test)
    print('Shape of Test3', df_test.shape)
    # OneHot encoding and Label encoding
    df_test =  category_to_numeric( df_test=df_test) 
    print('Shape of Test4', df_test.shape)
    # create custome features for train and test application
    df_test = create_custom_features_main_table(df_test=df_test)
    print('Shape of application test ', df_test.shape)
    # Handling missing value-  Category
    df_test=  replacing_missing_numeric(df_test = df_test)
    # Bureau table custom features
    df_bureau = create_custom_bureau_feature(df_bureau=df_bureau)
    # Join application and bureau table
    df_main_test = join_application_bureau(df = df_test, df_bureau=df_bureau)
    # Join application, bureau and Bureau_balance
    df_main_test = merge_application_BureauBal(df_main=df_main_test, df_bureau=df_bureau, df_bureau_bal=df_bureau_bal)
    #Creating custom features for previous applications
    df_prev_app = create_custom_prevapp_feature(previous_application=df_prev_app)
    # Join application, bureau , Bureau_balance and previous application
    df_main_test = merge_application_prev_app(df_main=df_main_test, df_prev_app=df_prev_app)  #-------->
    # create new feature based on existing column in POS table
    df_pos = create_custom_pos_feature(pos_cash=df_pos)
    # Join application, bureau , Bureau_balance , previous application and pos
    df_main_test = merge_application_pos(df_main=df_main_test, df_pos=df_pos)  
    #Creating custom features - Install payment features
    df_inst_pay  = create_custom_Inspay_feature(installments_payments=df_inst_pay)
    # Join application, bureau , Bureau_balance , previous application , pos and Installment payment
    df_main_test = merge_application_Inspay(df_main=df_main_test, df_inst_pay=df_inst_pay)  
    # Create new feature Creditcard balance
    df_credit_bal = create_custom_Credit_bal_feature(cc_balance=df_credit_bal)
    # Join application, bureau , Bureau_balance , previous application , pos , Installment payment and Credit balance
    df_main_test = merge_application_credit_bal(df_main=df_main_test, df_credit_bal=df_credit_bal)  
    
    return df_main_test
    

def final_function_2(main_preprocessor_data, y_test):
      '''
      Function 2 for prediction. This function takes both the Test Point and Target value of that point. It returns
      the prediction along with the metric for the predicted points.
                
      '''
      
      start_time = datetime.now()
      infile = open('select_features.txt','rb')
      selected_features = pickle.load(infile)  
      test_test = main_preprocessor_data[selected_features]

      # load best model
      # Saving the final model LightGBM as pickle file for the future use in productionizing the model
      with open('Best_lgbm.pkl','rb') as fp:
          best_model = pickle.load( fp)

      y_pred_prob = best_model.predict(test_test)

      y_pred = np.ones((len(test_test),), dtype=int)
      for i in range(len(y_pred_prob)):
          if y_pred_prob[i]<=0.5:
              y_pred[i]=0
          else:
              y_pred[i]=1

      ed = pd.DataFrame({'Actual':y_test[:test_test.shape[0]], 'Predicted': y_pred ,'prob':y_pred_prob})
      pd.set_option('display.max_rows', None)


      predicted_classes = np.where(y_pred_prob > 0.5, 1, 0)
      print(f"\nThe predicted class labels are:\n{predicted_classes}")
        
		
      return predicted_classes

      print(f"Total Time taken for prediction = {datetime.now() - start_time}")  	
		

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = request.form.values()
    int_features = [int(x) for x in request.form.values()]
    print(int_features[0])
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)
    # Read the Datatframe
    os.chdir(r'D:\C\User\Desktop\Applied_A\Assignment_All_in_one\Casestudy')
    df_main = pd.read_csv(r"D:\C\User\Desktop\Applied_A\Assignment_All_in_one\Casestudy\application_train.csv")

    #test_datapoint_func_2 = df_main.sample(1).copy()
    test_datapoint_func_2 = df_main[df_main['SK_ID_CURR'] == int_features[0]]
    targets_func_2 = test_datapoint_func_2.pop('TARGET')
    main_preprocessor_data = final_function_1(input_data = test_datapoint_func_2)
    output = final_function_2(main_preprocessor_data,targets_func_2)
    
    if int(output) == 0:
        return render_template('index.html', prediction_text='Customer having good Credit history, capabale to repay a debt ')

    else:
        return render_template('index.html', prediction_text='Customer having bad Credit history, risky customer ')

if __name__ == "__main__":
    app.run(debug=True)
