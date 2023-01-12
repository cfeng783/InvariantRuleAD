from .data_util import DataUtil

__all__ = ['DataUtil']

def train_val_split(df,val_ratio):
    """
    split the dataframe into a train part and a validation part
    
    Parameters
    ----------
    df : DataFrame
        The dataset
    val_ratio : float
        the proportion of the validation part
        
    Returns
    -------
    Dataframe 
        the training dataframe
    Dataframe 
        the validation dataframe
    """
    pos = int(len(df)*(1-val_ratio))
    val_df = df.loc[pos:,:]
    val_df = val_df.reset_index(drop=True)
    train_df = df.loc[:pos,:]
    train_df = train_df.reset_index(drop=True)
    return train_df,val_df