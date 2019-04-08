import torch
import numpy as np
from torch.utils.data import DataSet, DataLoader
from torch._six import int_classes as _int_classes
from fastai import DataBunch

class TFDataSet(Dataset):
    """Create a DataSet object for the NASA Turbofan Dataset"""

    def __init__(self, df):
        """
        Args:
            df (DataFrame): a pandas dataframe containing the Turbofan dataset (cleaned up). 
            The last column of the dataframe contains the independent variable we are trying to predict.
        """
        self.x = torch.from_numpy(df.iloc[:,:-1].values)
        self.y = torch.from_numpy(df.iloc[:,-1].values)
        #self.transform = transform --> No transform at this stage (but we should probably Normalize somewhere!!)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# The Turbofan dataset consists in sequences of data until failure for 100 different engine units.
# So, when we are going to sample sequence from the dataset, we MUST ensure that the sequence
# are not shuffled and that they DO NOT mix different engine units.

# I should implement a modified version of BatchSampler here instead
# and use SequentialSampler() into it. BatchSampler would need to ensure the
# different sample have a unique 'Unit' number, even if we lose data in the process

class SequenceSampler(Sampler):
    """Create a sequence sampler for the Turbofan dataset. Not sure this is useful... Can reuse SequentialSampler instead"""

    def __init__(self, data_source):
        """
        Args:
            (Dataset): dataset to sample from
        """
        self.data_source = data_source
    
    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)



#train_ds = TensorDataset(x_train, y_train)
#valid_ds = TensorDataset(x_valid, y_valid)
#data = DataBunch.create(train_ds, valid_ds, bs=bs)