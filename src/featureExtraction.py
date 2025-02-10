import spacy
import nltk
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from textblob import TextBlob

nltk.download("stopwords")
nlp = spacy.load("es_core_news_sm")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
stopwords = set(nltk.corpus.stopwords.words("spanish"))
sentimentAnalyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

class featureExtraction:

    def __init__(self, text):
        self.text = text
        self.doc = nlp(text)
        self.words = [token.text.lower() for token in self.doc if token.is_alpha]
        self.numWords = len(self.words)
        self.sentences = [sent.text for sent in self.doc.sents]
        

    def lexicalFeatures(self):
        numUniqueWords = len(set(self.words))
        lexicalDiversity = numUniqueWords / self.numWords if self.numWords > 0 else 0
        avgWordsLength = np.mean([len(word) for word in self.words]) if self.words else 0
        numPronouns = sum(1 for token in self.doc if token.pos_ == "PRON")
        functionalWords = sum(1 for token in self.doc if token.text.lower() in stopwords) / self.numWords if self.numWords > 0 else 0

        return {
            "lexical-diversity": lexicalDiversity,
            "average-word-length": avgWordsLength,
            "number-of-pronouns": numPronouns,
            "functional-words": functionalWords
        }
    
    def syntacticFeatures(self):
        numSentences = len(self.sentences)
        avgSentenceLength = self.numWords / numSentences if numSentences > 0 else 0

        verbTenses = Counter([token.tag_ for token in self.doc if token.pos == "VERB"])
        pastTenses = verbTenses.get("VMP", 0) / self.numWords if self.numWords > 0 else 0
        presentTenses = verbTenses.get("VMI", 0) / self.numWords if self.numWords > 0 else 0

        numSubordinating = sum(1 for token in self.doc if token.dep_ == "mark")
        subordinationRatio = numSubordinating / numSentences if numSentences > 0 else 0

        return {
            "average-sentence-length": avgSentenceLength,
            "past-tenses": pastTenses,
            "present-tenses": presentTenses,
            "subordination-ratio": subordinationRatio
        }
    

    def semanticFeatures(self):
        sentencesEmbeddings = embedder.encode(self.text).mean()

        coherenceScores = []
        for i in range(len(self.sentences) -1):
            vec1 = embedder.encode(self.sentences[i])
            vec2 = embedder.encode(self.sentences[i+1])
            similarity = np.dot(vec1, vec2) / ((np.linalg.norm(vec1)) * (np.linalg.norm(vec2)))
            coherenceScores.append(similarity)

        avgCoherence = np.mean(coherenceScores) if coherenceScores else 0
            
        wordFrequence = Counter(self.words)
        mostCommonWord, mostCommonCount = wordFrequence.most_common(1)[0] if wordFrequence else ("",0)
        repetitionRatio = mostCommonCount / self.numWords if self.numWords > 0 else 0

        return {
            "semantic-embedding-mean": sentencesEmbeddings,
            "average-coherence": avgCoherence,
            "repetition-ratio": repetitionRatio
        }
    

    def extractFeatures(self):
        features = {}
        features.update(self.lexicalFeatures())
        features.update(self.syntacticFeatures())
        features.update(self.semanticFeatures())
        features.update(self.sentimentAnalysis())
        return features
    

    def sentimentAnalysis(self):
        sentimentResult = sentimentAnalyzer(self.text)[0]
        sentimentLabel = sentimentResult["label"]
        sentimentScore = sentimentResult["score"]

        sentimentMapping = {
            "1 stars": 0,
            "2 stars": 1,
            "3 stars": 2,
            "4 stars": 3,
            "5 stars": 4
        }
        sentimentValue = sentimentMapping.get(sentimentLabel, 0)

        blob = TextBlob(self.text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        return {
            "sentiment-value": sentimentValue,
            "sentiment-score": sentimentScore,
            "polarity": polarity,
            "subjectivity": subjectivity
        }