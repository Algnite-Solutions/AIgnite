from torch.utils.data import Dataset
from datasets import load_dataset

class DocFinQADataset(Dataset):
    def __init__(self, split="test"):
        self.ds = load_dataset("kensho/DocFinQA")
        self.split = split
        
    def __len__(self):
        return len(self.ds[self.split])
    
    def __getitem__(self, idx):
        item = self.ds[self.split][idx]
        return {
            "Context": item["Context"],
            "Question": item["Question"],
            "Answer": item["Answer"],
            'Program': item['Program']
        }

# Example usage:
# from torch.utils.data import DataLoader
# dataset = DocFinQADataset(split="test")
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)