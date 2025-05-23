from bs4 import BeautifulSoup
from uuid import uuid4
from pathlib import Path
#new
from datetime import datetime, timezone, date, timedelta
import arxiv
import os
import requests
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import time
import google.generativeai as genai
import re
from urllib.parse import urlparse
from typing import List, Tuple
import requests
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import base64
from volcengine.visual.VisualService import VisualService
import json


date = datetime.now(timezone.utc).date()


client = arxiv.Client()
one_day = timedelta(days=1)
yesterday = date - one_day

exact_time = "0600"
today_str = date.strftime("%Y%m%d") + exact_time
yesterday_str = yesterday.strftime("%Y%m%d") + exact_time
print(today_str)
query = "cat:cs.* AND submittedDate:[" + yesterday_str + " TO " + today_str + "]"
#query = "cat:cs.* AND submittedDate:[202504190900 TO 202504191200]"
print(today_str,yesterday_str)

search = arxiv.Search(
    query=query,
    max_results=8,  # You can set max papers you want here
    sort_by=arxiv.SortCriterion.SubmittedDate
)

for r in client.results(search):
  print(r.title)