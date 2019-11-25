import re
import os
import math
import operator

from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from stemming.porter2 import stem
from collections import Counter

class NaiveBayesClassifier:
    def __init__(self):
        self.dev_dir = '' # 트레이닝데이터의 경로
        self.test_dir = '' # 테스트데이터의 경로
        self.categories = [] # 카테고리
        self.cat_text_names = [] # 카테고리별 파일이름 배열
        self.cat_texts = [] # 카테고리별 원문 전체 배열
        self.cat_stems = [] # 카테고리별 어간 배열
        self.cat_stems_count = [] # 카테고리별 (어간,중복횟수) 배열
        self.post_probs = [] # 카테고리별 사전확률
        self.cat_likelihoods = [] # 카테고리별 우도 (어간,우도) 배열
        self.smoothing_num = 0.5 # 평활화 상수
        self.sum_cat_prob = [] # 카테고리별 어간 수

        self.test_text_names = [] # 테스트데이터 텍스트파일 이름 배열
        self.test_texts = [] # 테스트데이터 텍스트파일별 원문
        self.test_stems = [] # 테스트데이터 텍스트파일별 어간 배열

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
            self.cat_text_names.append(text_names)
            # 각 카테고리 안의 텍스트 파일 불러오기
            crnt_text = ''
            for text_name in text_names:
                crnt_dir = self.dev_dir + category + '/' + text_name
                f = open(crnt_dir, 'r')
                crnt_text = crnt_text + f.read()
            # 최종적으로 카테고리별 텍스트 리스트 안에 뭉쳐진 텍스트가 들어감
            self.cat_texts.append(crnt_text)
        f.close()

    """트레이닝 데이터 Tokenize 및 Stemming 처리"""
    def dev_tokenize_stem(self):
        # 카테고리당 하나로 통합된 텍스트를 기준으로 처리
        for (index, original_text) in enumerate(self.cat_texts):
            self.cat_stems.append(self.tokenize_stems(original_text))

    """Tokenize 및 Stemming 범용 함수"""
    def tokenize_stems(self, original_text):
        sentences = []
        word_list = []
        temp = []
        # 문장단위 토큰화
        for line in sent_tokenize(original_text):
            sentences.append(line.replace("\n", " "))
        # TreeBankWord 이용 토큰화
        for sentence in sentences:
            tokens = TreebankWordTokenizer().tokenize(sentence)
            temp = temp + tokens
        word_list = temp
        temp = []
        # 토큰화된 배열 전처리
        for word in word_list:
            # 특수문자 제거
            word = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', word)
            # 소문자화
            word = word.lower()
            # 숫자 제거 후 저장
            if not word.isdigit() and not word == '':
                temp.append(word)
        word_list = temp
        temp = []
        # 어간 추출
        for word in word_list:
            temp.append(stem(word))
        word_list = temp
        temp = []
        # 불용어 제거
        stop_words = set(stopwords.words('english'))
        for word in word_list:
            if not word in stop_words:
                temp.append(word)
        word_list = temp
        temp = []
        # 최종적으로 처리된 단어들을 저장
        return word_list

    """어간 개수 세기"""
    def count_stems(self):
        # 카테고리별로 어간, 중복횟수 저장
        for stems in self.cat_stems:
            self.cat_stems_count.append(Counter(stems))

    """사전확률 계산"""
    def calc_post_probs(self):
        # 사전확률 : 각각의 학습데이터(문서)가 카테고리에 해당하는 비율
        sum_prob = 0
        for names in self.cat_text_names:
            sum_prob = sum_prob + len(names).__int__()
        for names in self.cat_text_names:
            self.post_probs.append(len(names)/sum_prob)

    """우도 계산"""
    def calc_likelihoods(self):
        # 우도 : 해당 단어가 해당 범주의 전체 단어 수 대비 몇 개나 되는가
        sum_cat_prob = 0
        # 카테고리별로 우도 계산
        for stem_counts in self.cat_stems_count:
            # 해당 카테고리의 전체 단어 개수 세기
            sum_cat_prob = 0
            temp = {}
            for index, (key, value) in enumerate(stem_counts.items()):
                sum_cat_prob += value
            self.sum_cat_prob.append(sum_cat_prob)
            # 해당 카테고리의 단어순으로 반복
            for index, (key, value) in enumerate(stem_counts.items()):
                # 평활화 상수를 각각 더해 0이 되는것을 방지함
                temp[key] = value + self.smoothing_num / sum_cat_prob + (self.smoothing_num * 2)
            self.cat_likelihoods.append(temp)

    """트레이닝 데이터 입력부터 우도 계산까지 일괄처리"""
    def train_data(self):
        self.load_dev_data()
        self.dev_tokenize_stem()
        self.count_stems()
        self.calc_post_probs()
        self.calc_likelihoods()

    """테스트 데이터 입력"""
    def load_test_data(self):
        test_data_names = os.listdir(self.test_dir)
        if ".DS_Store" in test_data_names:
            test_data_names.remove(".DS_Store")
        for test_data_name in test_data_names:
            self.test_text_names.append(test_data_name)
            crnt_dir = self.test_dir + test_data_name
            f = open(crnt_dir, 'r')
            self.test_texts.append(f.read())

    """테스트 데이터 Tokenize 및 Stemming 처리"""
    def test_tokenize_stem(self):
        # 각 원문에 대해 아래 작업
        for test_text in self.test_texts:
             self.test_stems.append(self.tokenize_stems(test_text))

    """우도를 기준으로 사후확률을 계산하여 어느 카테고리인지 분류한 후 결과 출력"""
    def classify_test_data(self):
        # 테스트데이터 파일별로 반복함
        for (test_index, test_stem) in enumerate(self.test_stems):
            result_probs = {}
            # 카테고리별로 반복함
            for (dev_index, post_prob) in enumerate(self.post_probs):
                # 해당 테스트데이터 파일이 해당 카테고리에 속할 사후확률
                category_prob = 1
                category_negative = 0
                # 테스트데이터 어간별로 반복함
                for test_each_stem in test_stem:
                    # 테스트데이터에서 가져온 어간이 현재 카테고리의 어간 목록에 있을 경우
                    if test_each_stem in self.cat_stems[dev_index]:
                        # 해당하는 우도를 누적하여 곱한다
                        category_prob *= self.cat_likelihoods[dev_index][test_each_stem]
                    else:
                        # 해당하지 않는 경우, 부정 확률 ( 0 / 카테고리 전체 어간 수 ), 평활화 적용
                        category_prob *= self.smoothing_num / self.sum_cat_prob[dev_index] + (self.smoothing_num * 2)
                # 우도 누적이 끝나면, 사전확률 * 우도 누적곱으로 확률 계산
                result_probs[self.categories[dev_index]] = post_prob * category_prob
            # 결과 출력
            print( '*' + self.test_text_names[test_index] + ' 파일에 대한 결과 ')
            result = sorted(result_probs.items(), key=operator.itemgetter(1), reverse=True)
            for index, (key, value) in enumerate(result):
                print('[',key,']', value, end='')
            print('\n')

    """테스트 데이터 분류 일괄처리"""
    def classify(self):
        self.load_test_data()
        self.test_tokenize_stem()
        self.classify_test_data()

NBC = NaiveBayesClassifier()
NBC.set_dir("./dev/", "./test/")
NBC.train_data()
NBC.classify()