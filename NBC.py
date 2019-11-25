import re
import os
import nltk
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize

class NaiveBayesClassifier:
    def __init__(self):
        self.dev_dir = ''
        self.test_dir = ''
        self.categories = []
        self.cat_texts = []

    """트레이닝, 테스트 데이터 경로설정"""
    def set_dir(self, dev_dir, test_dir):
        self.dev_dir = dev_dir
        self.test_dir = test_dir

    """파일로부터 원본 텍스트 불러오기"""
    def load_dev_data(self):
        # 카테고리 불러오기
        self.categories = os.listdir(self.dev_dir)
        # Mac OS용 DS_store 삭제
        if ".DS_Store" in self.categories:
            self.categories.remove(".DS_Store")
        # 각 카테고리의 파일명 불러오기
        for category in self.categories:
            text_names = os.listdir(self.dev_dir + category)
            # 각 카테고리 안의 텍스트 파일 불러오기
            crnt_text = ''
            for text_name in text_names:
                crnt_dir = self.dev_dir + category + '/' + text_name
                f = open(crnt_dir, 'r')
                crnt_text = crnt_text + f.read()
            # 최종적으로 카테고리별 텍스트 리스트 안에 뭉쳐진 텍스트가 들어감
            self.cat_texts.append(crnt_text)

    """Tokenize 및 Stemming 처리"""
    def tokenize(self):
        for (index, original_text) in enumerate(self.cat_texts):
            sentences = []
            temp = []
            # 문장단위 토큰화
            for tokenized in sent_tokenize(original_text):
                sentences.append(tokenized.replace("\n", " "))
            # regexp 토큰화
            print(sentences)
            for sentence in sentences:
                tokens = TreebankWordTokenizer().tokenize(sentence)
                #tokens = regexp_tokenize(sentence,"[\w]+")
                temp = temp + tokens
            print(temp)
            # # 토큰화된 배열 전처리
            # for sentence in sentences:
            #     # 특수문자 제거
            #     sentence = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', sentence)
            #     # 소문자화
            #     sentence = sentence.lower()
            #     # 숫자 제거 후 저장
            #     if not sentence.isdigit() and not sentence == '':
            #         tokens.append(sentence)
            # sentences = tokens

NBC = NaiveBayesClassifier()
NBC.set_dir("./dev/", "./test/")
NBC.load_dev_data()
NBC.tokenize()