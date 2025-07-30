# XML scraping for xml file

import xml.etree.ElementTree as ET
import re, string, unicodedata
import nltk
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

# Path to XML file
file_path = r"C:\Users\anish\Data Science with Gen AI\PROJECTS\NLP WEB SCRAPING\xml_single articles\769952.xml"

# Parse XML
tree = ET.parse(file_path)
root = tree.getroot()

# Convert XML to string
xml_string = ET.tostring(root, encoding='utf8').decode('utf8')

# Text cleaning functions
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub(r'\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = re.sub('  ', '', text)
    return text

# Cleaned sample
sample = denoise_text(xml_string)
# print(sample)  # Uncomment to view cleaned text
