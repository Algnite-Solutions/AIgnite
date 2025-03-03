from .novelqa_dataloader import NovelQADataLoader

class NovelQADataset:
    def __init__(self, bookpath, datapath, metadatapath, book_ids=None):
        """
        Initialize the NovelQA dataset.
        Args:
            bookpath (str): Path to the books directory
            datapath (str): Path to the QA data directory
            metadatapath (str): Path to the book metadata file
            book_ids (list): List of book IDs to load. If None, loads all books.
        """
        self.bookpath = bookpath
        self.datapath = datapath
        self.metadatapath = metadatapath
        self.book_ids = book_ids
        
        # Load all data
        self.data = self._load_data()
        self.contexts = self._load_contexts()
    
    def _load_data(self):
        """Load QA data for all specified books"""
        data = {}
        loader = NovelQADataLoader(
            bookpath=self.bookpath,
            datapath=self.datapath,
            metadatapath=self.metadatapath
        )
        
        # If no book_ids specified, get all available books
        if self.book_ids is None:
            # You might need to implement a method to get all available book IDs
            raise NotImplementedError("Please specify book_ids or implement method to get all books")
            
        for bid in self.book_ids:
            data[bid] = loader.get_data_from_a_book(bid=bid)
        
        return data
    
    def _load_contexts(self):
        """Load context for all specified books"""
        contexts = {}
        loader = NovelQADataLoader(
            bookpath=self.bookpath,
            datapath=self.datapath,
            metadatapath=self.metadatapath
        )
        
        for bid in self.book_ids:
            contexts[bid] = loader.get_content_from_a_book(bid=bid)
        
        return contexts
    
    def __getitem__(self, idx):
        """Get a single QA pair"""
        # Convert flat index to (book_id, qa_pair_id)
        book_id = self._index_to_book_id(idx)
        qa_pair = self._get_qa_pair(book_id, idx)
        
        return {
            'book_id': book_id,
            'Question': qa_pair['Question'],
            'Answer': qa_pair['Answer'],
            'Context': self.contexts[book_id]
        }
    
    def _index_to_book_id(self, idx):
        """Convert flat index to book_id"""
        current_count = 0
        for bid, book_data in self.data.items():
            if current_count + len(book_data) > idx:
                return bid
            current_count += len(book_data)
        raise IndexError("Index out of range")
    
    def _get_qa_pair(self, book_id, global_idx):
        """Get QA pair from book using global index"""
        current_count = 0
        for bid, book_data in self.data.items():
            if bid == book_id:
                local_idx = global_idx - current_count
                return list(book_data.values())[local_idx]
            current_count += len(book_data)
    
    def __len__(self):
        """Total number of QA pairs across all books"""
        return sum(len(book_data) for book_data in self.data.values())

