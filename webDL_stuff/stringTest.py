"""
#below finds an index of a word in a string

import re

text = "This is an example sentence."
word = "example"
#returns index of first letter of word
index = text.find(word)
if index != -1:
    print(f"First occurrence of '{word}' at index {index}.")
else:
    print(f"'{word}' not found in the text.")
    """
from bs4 import BeautifulSoup
import requests

#below finds a URL to steal and trim

URL = "https://learn.microsoft.com/en-us/windows/win32/api/objectarray/"
page = requests.get(URL)
if(page.ok):
    soup = BeautifulSoup(page.content, "lxml")
    infoDiv = soup.get_text()
    
    startIndex = infoDiv.find("article")
    print(startIndex)
    endIndex = infoDiv.find("Was this page helpful")
    print(endIndex)

    
    #text = repr()
    print(infoDiv[startIndex:endIndex])

