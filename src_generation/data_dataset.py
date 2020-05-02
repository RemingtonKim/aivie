import torch.utils.data as data

class DatasetAB(data.Dataset):
    """Dataset used for images in class A and class B"""
    
    def __init__(self, *datasets) -> None:
        """Creates DatasetAB instance"""
        self.datasets = datasets
    
    def __len__(self) -> int:
        """Returns length of max dataset"""
        length = [len(d) for d in self.datasets]

        return max(length)

    def __getitem__(self, index:int) -> tuple:
        """Returns an element from dataset A and B"""
        d = [dataset for dataset in self.datasets]
        assert len(d) == 2

        a = d[0][index%len(d[0])]
        b = d[1][index%len(d[1])]
        
        return a[0], b[0]
