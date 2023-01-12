import pandas as pd
import numpy as np
from .signals import ContinuousSignal,CategoricalSignal
import zipfile
from enum import Enum

class DATASET(Enum):
    SWAT = 101
    BATADAL = 102
    KDDCup99 = 103
    GasPipeline = 104
    Annthyroid = 105
    Cardio = 106
    

def load_dataset(ds):
    if ds == DATASET.SWAT:
        return load_swat_data()
    elif ds == DATASET.KDDCup99:
        return load_kddcup99_10_percent_data()
    elif ds == DATASET.GasPipeline:
        return load_gas_data()
    elif ds == DATASET.Annthyroid:
        return load_annthyroid_data()
    elif ds == DATASET.Cardio:
        return load_cardio_data()
    elif ds == DATASET.BATADAL:
        return load_batadal_data()
    else:
        print('Cannot find the dataset!')
        return None


def load_swat_data(fp='../datasets/SWAT'):
    """
    load SWAT dataset
    
    Parameters
    ----------
    fp : string, default is '../datasets/SWAT'
        the path of the folder where the dataset locates
        
    Returns
    -------
    Dataframe 
        the training dataframe
    
    Dataframe 
        the testing dataframe
    
    ndarray 
        the test labels
    
    list 
        the list of signals
         
    """
    
    z_tr = zipfile.ZipFile(fp+'/SWaT_train.zip', "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    train_df=pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()
    
    z_tr = zipfile.ZipFile(fp+'/SWaT_test.zip', "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    test_df=pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()
    
    test_df['label'] = 0
    test_df.loc[test_df['Normal/Attack']!='Normal', 'label'] = 1
    
    

    sensors = ['FIT101','LIT101','AIT201','AIT202','AIT203','FIT201',
           'DPIT301','FIT301','LIT301','AIT401','AIT402','FIT401',
           'LIT401','AIT501','AIT502','AIT503','AIT504','FIT501',
           'FIT502','FIT503','FIT504','PIT501','PIT502','PIT503','FIT601',]

    actuators = ['MV101','P101','P102','MV201','P201','P202',
                   'P203','P204','P205','P206','MV301','MV302',
                   'MV303','MV304','P301','P302','P401','P402',
                   'P403','P404','UV401','P501','P502','P601',
                   'P602','P603']
    
    for sensor in sensors:
        train_df['delta_'+sensor] = train_df[sensor].shift(-1)-train_df[sensor]
        test_df['delta_'+sensor] = test_df[sensor].shift(-1)-test_df[sensor]
    
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    
    signals = []
    for name in sensors:
        if train_df[name].min() != train_df[name].max():
            signals.append( ContinuousSignal(name, isInput=True, isOutput=True, 
                                        min_value=train_df[name].min(), max_value=train_df[name].max(),
                                        mean_value=train_df[name].mean(), std_value=train_df[name].std()) )
            signals.append( ContinuousSignal('delta_'+name, isInput=True, isOutput=True, 
                                        min_value=train_df['delta_'+name].min(), max_value=train_df['delta_'+name].max(),
                                        mean_value=train_df['delta_'+name].mean(), std_value=train_df['delta_'+name].std()) )
            
    for name in actuators:
        if len(train_df[name].unique().tolist()) >= 2:
            signals.append( CategoricalSignal(name, isInput=True, isOutput=True, 
                                                values=train_df[name].unique().tolist()) )

    return train_df,test_df,test_df['label'].values,signals

def load_batadal_data(fp='../datasets/BATADAL/'):
    """
    load DATADAL dataset
    """
    
    z_tr = zipfile.ZipFile(fp+'/BATADAL_dataset03.zip', "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    train_df=pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()
    
    z_tr = zipfile.ZipFile(fp+'/BATADAL_dataset04.zip', "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    test_df=pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()
    
    new_cols = {}
    for col in test_df.columns:
        new_cols[col]=col.strip()
    
    test_df.rename(columns = new_cols, inplace = True)
    
    test_df['label'] = 0
    test_df.loc[test_df['ATT_FLAG'] != -999, 'label'] = 1
    signals = []
    for name in train_df:
        if name!='DATETIME' and name!='ATT_FLAG' and len(train_df[name].unique())>5:
            signals.append(ContinuousSignal(name, isInput=True, isOutput=True,
                                            min_value=train_df[name].min(), max_value=train_df[name].max(),
                                            mean_value=train_df[name].mean(), std_value=train_df[name].std()))
        elif name!='DATETIME' and name!='ATT_FLAG' and len(train_df[name].unique())<=5:
            signals.append( CategoricalSignal(name, isInput=True, isOutput=False, 
                                                values=train_df[name].unique().tolist()) ) 
            
    return train_df, test_df, test_df['label'].values, signals



def load_kddcup99_10_percent_data(fp='../datasets/kddcup99/'):
    """
    load kddcup99_10_percent dataset

    Parameters
    ----------
    fp : string, default is '../datasets/categorical data/'
        the path of the folder where the dataset locates

    Returns
    -------
    Dataframe
        the training dataframe

    Dataframe
        the testing dataframe

    ndarray
        the test labels

    list
        the list of signals

    -------

        """    
    z_tr = zipfile.ZipFile(fp+'/kddcup_data_10_percent_normal_train.zip', "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    train_df=pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()
    
    z_tr = zipfile.ZipFile(fp+'/kddcup_data_10_percent_test.zip', "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    test_df=pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()

    test_df['label'] = 0
    test_df.loc[test_df['Normal/Attack'] != 'normal.', 'label'] = 1
    pos_df = test_df.loc[test_df['label']==0,:]
    neg_df = test_df.loc[test_df['label']==1,:]
    neg_df = neg_df.sample(n=len(pos_df)//4)
    test_df = pd.concat([pos_df,neg_df])
    test_df = test_df.reset_index()


    sensors = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot',
               'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
               'num_shells', 'num_access_files', 'num_outbound_cmds', 'count', 'srv_count', 'serror_rate',
               'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
               'dst_host_srv_count', 'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
               'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate', 'dst_host_srv_rerror_rate',]

    actuators = ['protocol_type','service','flag','land','logged_in','is_host_login','is_guest_login',]
    signals = []
    for name in sensors:
        if train_df[name].min() != train_df[name].max():
            signals.append(ContinuousSignal(name, isInput=True, isOutput=True,
                                            min_value=train_df[name].min(), max_value=train_df[name].max(),
                                            mean_value=train_df[name].mean(), std_value=train_df[name].std()))
    for name in actuators:
        # train_df.loc[train_df[name]==2,name]=3
        # test_df.loc[test_df[name]==2,name]=3

        if len(train_df[name].unique().tolist()) >= 2:
            signals.append(CategoricalSignal(name, isInput=True, isOutput=False,
                                             values=train_df[name].unique().tolist()))
        # elif len(train_df[name].unique().tolist()) == 2:
        #     signals.append( BinarySignal(name, SignalSource.controller, isInput=True, isOutput=False,
        #                                         values=train_df[name].unique().tolist()) )

    return train_df.loc[:, sensors + actuators], test_df.loc[:, sensors + actuators], test_df['label'].values, signals

def load_gas_data(fp='../datasets/'):
    
    z_tr = zipfile.ZipFile(fp+'/IanArffDataset.zip', "r")
    f_tr = z_tr.open(z_tr.namelist()[0])
    df=pd.read_csv(f_tr)
    f_tr.close()
    z_tr.close()
    
    df = df.replace('?',np.nan)
    df = df.fillna(method='ffill')
    df = df.apply(pd.to_numeric)
    df['interval'] = df['time'].shift(-1)-df['time']
    df = df.dropna()
    
    df.drop(['specific result','categorized result','time'], axis = 1, inplace = True)
    pos = len(df)*3//5
    
    train_df = df.loc[:pos,:].reset_index(drop=True)
    train_df = train_df.loc[train_df['binary result']==0,:].reset_index(drop=True)
    
    test_df = df.loc[pos:,:].reset_index(drop=True)
    pos_df = test_df.loc[test_df['binary result']==0,:]
    neg_df = test_df.loc[test_df['binary result']==1,:]
    neg_df = neg_df.sample(n=len(pos_df)//4)
    test_df = pd.concat([pos_df,neg_df])
    
    
    signals = []
    for name in train_df:
        if name not in ['binary result',] and len(train_df[name].unique())>5:
            signals.append( ContinuousSignal(name, isInput=True, isOutput=True, 
                                            min_value=train_df[name].min(), max_value=train_df[name].max(),
                                            mean_value=train_df[name].mean(), std_value=train_df[name].std() ) ) 
        elif name not in ['binary result',] and len(train_df[name].unique())<=5:
            signals.append( CategoricalSignal(name, isInput=True, isOutput=False, 
                                                values=train_df[name].unique().tolist()) ) 
    
    return train_df,test_df,test_df['binary result'].values,signals

def load_annthyroid_data(fp='../datasets/'):
    """
    load annthyroid dataset
    """
    df = pd.read_csv(fp+'annthyroid_21feat_normalised.csv')
     
    pos = len(df)*3//5
    
    train_df = df.loc[:pos,:].reset_index(drop=True)
    train_df = train_df.loc[df['class']==0,:].reset_index(drop=True)
    
    test_df = df.loc[pos:,:].reset_index(drop=True)
    
    signals = []
    for name in train_df:
        if name!='class' and len(train_df[name].unique())>5:
            signals.append( ContinuousSignal(name, isInput=True, isOutput=True, 
                                            min_value=train_df[name].min(), max_value=train_df[name].max(),
                                            mean_value=train_df[name].mean(), std_value=train_df[name].std() ) ) 
        elif name!='class' and len(train_df[name].unique())<=5:
            signals.append( CategoricalSignal(name, isInput=True, isOutput=False, 
                                                values=train_df[name].unique().tolist()) ) 
    return train_df,test_df,test_df['class'].values,signals

def load_cardio_data():
    """
    load npz dataset
    """
    data = np.load('../datasets/cardio.npz', allow_pickle=True) 
    
    X = data['X']
    y = data['y']
    
    cols = []
    for i in range(X.shape[1]):
        cols.append('col'+str(i))
    df = pd.DataFrame(data=X,columns=cols)
    
    df['class'] = y
    
    pos = len(df)*3//5
    
    train_df = df.loc[:pos,:].reset_index(drop=True)
    train_df = train_df.loc[df['class']==0,:].reset_index(drop=True)
    
    test_df = df.loc[pos:,:].reset_index(drop=True)
    
    pos_df = test_df.loc[test_df['class']==0,:]
    neg_df = test_df.loc[test_df['class']==1,:]
    
    if len(neg_df) > len(pos_df)//4:
        neg_df = neg_df.sample(n=len(pos_df)//4)
        test_df = pd.concat([pos_df,neg_df])

    signals = []
    for name in train_df:
        if name!='class' and len(train_df[name].unique())>5:
            signals.append( ContinuousSignal(name, isInput=True, isOutput=True, 
                                            min_value=train_df[name].min(), max_value=train_df[name].max(),
                                            mean_value=train_df[name].mean(), std_value=train_df[name].std() ) )
        elif name!='class' and len(train_df[name].unique())<=5:
            signals.append( CategoricalSignal(name, isInput=True, isOutput=False, 
                                                values=train_df[name].unique().tolist()) )
    return train_df,test_df,test_df['class'].values,signals

