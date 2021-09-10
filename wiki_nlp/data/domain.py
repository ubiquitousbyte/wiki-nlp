from typing import List
from dataclasses import dataclass

@dataclass 
class Paragraph:
    id: str 
    text: str

@dataclass 
class Section:
    id: str 
    title: str 
    position: str 
    paragraphs: List[Paragraph]

    def __post_init__(self):
        if len(self.paragraphs) > 0 and type(self.paragraphs[0]) == dict:
            self.paragraphs = [Paragraph(**paragraph) for paragraph in self.paragraphs]

@dataclass 
class Document:
    id: str 
    title: str 
    excerpt: str 
    source: str 
    sections: List[Section]

    def __post_init__(self):
        if len(self.sections) > 0 and type(self.sections[0]) == dict:
            self.sections = [Section(**section) for section in self.sections]

@dataclass 
class Category:
    id: str 
    name: str 
    subcategories: List['Category']
    documents: List[Document]