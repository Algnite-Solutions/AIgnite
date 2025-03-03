"""Dataloader class for the NovelQA dataset: for loding data from various formats.

Examples:
```
from dataloader import NovelQADataLoader
dataloader = NovelQADataloader(bookpath=<yourbookfolder>, datapath=<yourdatafolder>, metadatapath=<yourmetadatafile>)

# Get the NovelQA dataset. This function returns value that satiesfies the most basic testings.
dataloader.get_dataset_wo_metadata()

# Get the NovelQA dataset with metadata embedded.
dataloader.get_dataset_w_metadata()      

# Get the book's content with a bood id.
dataloader.get_content_from_a_book(bid="B00")

# Get the QA data of a book, with the book id.
dataloader.get_data_from_a_book(bid="B00")

# Get truncated book content. The range can either be percentatges or concrete #tokens.
dataloader.truncate_a_book_range(
                            bid="B00", 
                            position_range=[0,20], 
                            is_percentage=True,
                            ensure_sentence=True,
                            tokenizer=GPT2TokenizerFast.from_pretrained('Xenova/gpt-3.5-turbo-16k')
                            )
dataloader.truncate_a_book_range(
                            bid="B00", 
                            position_range=[0,128000], 
                            is_percentage=False,
                            ensure_sentence=True,
                            tokenizer=GPT2TokenizerFast.from_pretrained('Xenova/gpt-3.5-turbo-16k')
                            )

# Get the dataset with books truncated among a specific range, without metadata. 
dataloader.truncate_books_range_wo_metadata(
                            position_range=[0,20], 
                            is_percentage=True,
                            ensure_sentence=True,
                            tokenizer=GPT2TokenizerFast.from_pretrained('Xenova/gpt-3.5-turbo-16k')
                            )

# Get the dataset with books truncated among a specific range, with metadata. 
dataloader.truncate_books_range_w_metadata( 
                            position_range=[0,20], 
                            is_percentage=True,
                            ensure_sentence=True,
                            tokenizer=GPT2TokenizerFast.from_pretrained('Xenova/gpt-3.5-turbo-16k')
                            )

# Get a book's token count.
dataloader.get_book_token(bid="B00")

# Get the copyright status of a book.
dataloader.is_book_copyright(bid="B00")
```

# Write a modified-version NovelQA dataset to a path.
dataloader.write_dataset_to_files(dataset=dataloader._dataset, datapath=".")

"""
import json
import os
import random
import re

from transformers import GPT2TokenizerFast, PreTrainedTokenizerBase

tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/gpt-3.5-turbo-16k')


class NovelQADataLoader:
    """The dataloader class for loading novelQA data."""
    def __init__(self, 
                 bookpath:str = "CollectedBooks", 
                 datapath:str = "CollectedData", 
                 metadatapath:str = "bookmeta.json"
                 ) -> None:
        """The dataloader initialization function.

        Args:
            bookpath        : books' (txt's) folder path.
            datapath        : QA data's (json's) folder path.
            metadatapath    : path for book's metadata.
        """
        # Read book's metadata
        with open(metadatapath, "r") as infile:
            self._metadata = json.loads(infile.read())

        # Build dataset, expected format
        self._dataset = self.build_dataset(datapath=datapath, bookpath=bookpath)

    def build_dataset(self, bookpath:str, datapath:str):
        """Build the dataset.
        
        Args
            bookpath    :
            datapath    : 

        Returns:
            dataset     :
        """
        dataset = {}
        for root, dirs, files in os.walk(bookpath, topdown=True):
            for directory in dirs:   # Copyright Protected and Public Domain
                for filename in os.listdir(os.path.join(bookpath, directory)):
                    with open(os.path.join(bookpath, directory, filename), "r") as infile:
                        dataset[filename[:-4]] = {}
                        dataset[filename[:-4]]["book"] = infile.read()
        for root, dirs, files in os.walk(datapath, topdown=True):
            for directory in dirs:   # Copyright Protected and Public Domain
                for filename in os.listdir(os.path.join(datapath, directory)):
                    with open(os.path.join(datapath, directory, filename), "r") as infile:
                        if filename[:-5] in dataset:
                            dataset[filename[:-5]]["data"] = json.loads(infile.read())
        return dataset

    def get_dataset_wo_metadata(self) -> dict:
        """Get the NovelQA dataset, without each book's metadata. This should be mostly commonly used for usual evaluation.
        
        Returns:
            dataset : the dataset without book's metadata
        """
        return self._dataset

    def get_dataset_w_metadata(self) -> dict:
        """Get the NovelQA dataset, with each book's data containing the metadata. Usually for furthur analysis.
        
        Returns:
            dataset : the dataset with book's metadata
        """
        return {bid: {"metadata": self._metadata[bid], 
                "book": self._dataset[bid]["book"], 
                "data": self._dataset[bid]["data"]
                } for bid in self._dataset.keys()}

    def booktitle2bid(self, booktitle) -> str:
        """Get a bid with a book title. (legacy usage)"""
        def regular_title(title):
            if title=="Les Misérables":
                return "lesmiserables"
            return re.sub(r'[^a-z]', '', title.lower())
        for key, value in self._metadata.items():
            if regular_title(value["title"]) == regular_title(booktitle):
                return key
        raise KeyError

    def get_content_from_a_book(self, bid: str) -> str:
        """Get a specific book's content.

        Args:
            bid     : book's id

        Returns:
            content : book's content
        """
        return self._dataset[bid]["book"]

    def get_data_from_a_book(self, bid: str) -> dict:
        """Get the data from a specific book, with metadata.

        Args:
            bid     : book's id

        Returns:
            data    : the data specifically related to a book, with its metadata
        """
        return self._dataset[bid]["data"]

    def truncate_a_book_range(self, 
                              bid: str, 
                              position_range: list, 
                              is_percentage: bool = True, 
                              ensure_sentence: bool = True, 
                              tokenizer: PreTrainedTokenizerBase = GPT2TokenizerFast.from_pretrained('Xenova/gpt-3.5-turbo-16k')
                              ) -> str:
        """Get the truncated books within a token range.

        Args:
            bid             : book's id
            position_range  : trunction range, length 2 allowed, the start and end position/token of truncation.
            is_percentage   : True for the range indicates a percentage, False for the range indicates the specific tokens.
            ensure_sentence : True to ensure the truncation do not break sentences.
            tokenizer       : default to be GPT-3.5-turbo. 

        Returns:
            book_truncated  : truncated book content
        """
        # Tokenized book
        tokenized_book = [tokenizer.decode(t) for t in tokenizer.encode(self._dataset[bid]["book"])]
        
        # If is percentage, turn to token range
        if (len(position_range) != 2) or (not isinstance(position_range[0], (int, float))) or (not isinstance(position_range[1], (int, float))):
            print("Unexpected type of argument `positional_range`.")

        book_length = self._metadata[bid]["tokenlen"]

        chunk_range = position_range if not is_percentage else [int(book_length*i*0.01) for i in position_range]

        book_content = tokenized_book[chunk_range[0]:chunk_range[1]]
        book_content = "".join(book_content)

        # If ensure sentence, ensure sentences within the given range
        if ensure_sentence:
            abbreviations = {
            "Mr.": "Mr_PLACEHOLDER", 
            "Mrs.": "Mrs_PLACEHOLDER", 
            "Dr.": "Dr_PLACEHOLDER"
            }
            for abbr, placeholder in abbreviations.items():
                book_content = book_content.replace(abbr, placeholder)
            sentences = [sentence for sentence in re.split(r'(?<=[.!?]) +', book_content)]
            sentences = [sentence for sentence in map(lambda s: s.replace("_PLACEHOLDER", "."), sentences)]
            # find the first and last sentence mark and only preserve the contents inside
            if (tokenized_book[chunk_range[0]-1] in [".", "!", "?"]):
                sentences = sentences[:-1]
            else:
                sentences = sentences[1:-1]
            book_content = "".join(sentences)

        # Return the book content
        return book_content
    
    def truncate_books_range_wo_metadata(self, 
                                        position_range: list, 
                                        is_percentage: bool = True, 
                                        ensure_sentence: bool = True, 
                                        tokenizer: PreTrainedTokenizerBase = GPT2TokenizerFast.from_pretrained('Xenova/gpt-3.5-turbo-16k')
                                        ) -> dict:
        """Get the full dataset with truncated book without metadata.

        Args:
            bid             : book's id
            position_range  : trunction range, length 2 allowed, the start and end position/token of truncation.
            is_percentage   : True for the range indicates a percentage, False for the range indicates the specific tokens.
            ensure_sentence : True to ensure the truncation do not break sentences.
            tokenizer       : default to be GPT-3.5-turbo. 

        Returns:
            data_truncated  : dataset with truncated books without metadata
        """
        dataset = {}
        for bid, bdata in self._dataset.items():
            dataset[bid] = {}
            dataset[bid]["book"] = self.truncate_a_book_range(bid=bid, position_range=position_range, is_percentage=is_percentage, ensure_sentence=ensure_sentence, tokenizer=tokenizer)
            dataset[bid]["data"] = bdata["data"]

        return dataset

    def truncate_books_range_w_metadata(self, 
                                        position_range: list, 
                                        is_percentage: bool = True, 
                                        ensure_sentence: bool = True, 
                                        tokenizer: PreTrainedTokenizerBase = GPT2TokenizerFast.from_pretrained('Xenova/gpt-3.5-turbo-16k')
                                        ) -> dict:
        """Get the full dataset with truncated book with metadata.

        Args:
            bid             : book's id
            position_range  : trunction range, length 2 allowed, the start and end position/token of truncation.
            is_percentage   : True for the range indicates a percentage, False for the range indicates the specific tokens.
            ensure_sentence : True to ensure the truncation do not break sentences.
            tokenizer       : default to be GPT-3.5-turbo. 

        Returns:
            data_truncated  : dataset with truncated books with metadata
        """
        dataset = self.truncate_books_range_wo_metadata(
                    position_range=position_range, 
                    is_percentage=is_percentage,
                    ensure_sentence=ensure_sentence,
                    tokenizer=tokenizer
                    )
        return {bid: {"metadata": self._metadata[bid], 
                "book": dataset[bid]["book"], 
                "data": dataset[bid]["data"]
                } for bid in dataset.keys()}

    def get_book_token(self, bid: str, tokenizer: PreTrainedTokenizerBase = GPT2TokenizerFast.from_pretrained('Xenova/gpt-3.5-turbo-16k')) -> int:
        """Get the token count for a specific book.

        Args:
            bid         : book's id
            tokenizer   : default to be GPT-3.5-turbo. 

        Returns:
            book_token  : book's token count
        """
        return int(self._metadata[bid]["tokenlen"])

    def is_book_copyright(self, bid) -> bool:
        """Get the copyright status of a given book id.
        
        Args:
            bid             : the book id.
        
        Returns:
            status          : the copyright status of the given bid, true if copyright protected.
        """
        return True if self._metadata[bid]["copyright"] == "CopyrightProtected" else False

    def write_dataset_to_files(self, dataset, datapath: str = ".") -> None:
        """Write a dataset of the same dictionary-hierarchy as this class indicates to a given path. 
        
        This function will create a new folder `NovelQA` and write the data inside.

        Args:
            datapath        : path to write the NovelQA dataset. 
        """
        print(f"Begin writing dataset NovelQA to path {datapath}.")
        
        # Make paths
        os.mkdir(os.path.join(datapath, "NovelQA"))
        os.mkdir(os.path.join(datapath, "NovelQA", "Books"))
        os.mkdir(os.path.join(datapath, "NovelQA", "Data"))
        os.mkdir(os.path.join(datapath, "NovelQA", "Books", "PublicDomain"))
        os.mkdir(os.path.join(datapath, "NovelQA", "Books", "CopyrightProtected"))
        os.mkdir(os.path.join(datapath, "NovelQA", "Data", "PublicDomain"))
        os.mkdir(os.path.join(datapath, "NovelQA", "Data", "CopyrightProtected"))

        # Write data
        for bid, bdata in dataset.items():
            with open(os.path.join(datapath, "NovelQA", "Books", "CopyrightProtected" if self.is_book_copyright(bid) else "PublicDomain", bid+".txt"), "w", encoding="utf-8") as outfile:
                outfile.write(bdata["book"])
            with open(os.path.join(datapath, "NovelQA", "Data", "CopyrightProtected" if self.is_book_copyright(bid) else "PublicDomain", bid+".json"), "w", encoding="utf-8") as outfile:
                outfile.write(json.dumps(bdata["data"], indent=4, ensure_ascii=False))

        print(f"Finished writing dataset NovelQA to path {datapath}!")
        return
