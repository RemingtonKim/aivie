import torch.utils.data as data

class DatasetAB(data.Dataset):
    """Dataset used for images in class A and class B"""
    
    def __init__(self, *datasets) -> None:
        """Creates DatasetAB instance"""
        self.datasets = datasets
    
    def __len__(self) -> int:
        """Returns length of dataset, making sure that A and B are the same length"""
        length = len(set([len(d) for d in self.datasets]))

        assert length == 1
        return len(self.datasets[0])

    def __getitem__(self, index:int) -> tuple:
        """Returns an element from dataset A and B"""
        return tuple(d[index][0] for d in self.datasets)
