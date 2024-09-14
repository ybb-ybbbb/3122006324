import sys
import jieba
import unittest
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def remove_punctuation(text):
    """移除文本中的标点符号"""
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    return text

def tokenize_and_remove_punctuation(text):
    """中文分词并移除标点符号"""
    words = jieba.lcut(text)
    words = [remove_punctuation(word) for word in words]
    return words

def concatenate_words(words):
    """将分词后的单词列表重新组合为字符串"""
    return ' '.join(words)

def calculate_similarity(original_file, plagiarized_file):
    """计算相似度"""
    with open(original_file, 'r', encoding='utf-8') as f1, open(plagiarized_file, 'r', encoding='utf-8') as f2:
        original_text = f1.read()
        plagiarized_text = f2.read()

    # 中文分词并移除标点符号
    original_words = tokenize_and_remove_punctuation(original_text)
    plagiarized_words = tokenize_and_remove_punctuation(plagiarized_text)

    # 将分词后的文本重新组合为字符串
    original_str = concatenate_words(original_words)
    plagiarized_str = concatenate_words(plagiarized_words)

    # 计算 TF-IDF 特征
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([original_str, plagiarized_str])

    # 计算余弦相似度
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    # 计算相似度百分比
    similarity = cosine_sim * 100
    return similarity

def main():
    original_file = sys.argv[1]
    plagiarized_file = sys.argv[2]
    output_file = sys.argv[3]

    # 计算相似度
    similarity = calculate_similarity(original_file, plagiarized_file)

    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("{文件%s和文件%s的相似度为:%.2f}\n"% (original_file,plagiarized_file,similarity))



class PlagiarismDetectionTestCase(unittest.TestCase):
    def test_similarity_calculation(self):
        # 测试相似度计算函数的准确性

        original_file = 'E:\桌面\测试文本\orig.txt'
        plagiarized_file = 'E:\桌面\测试文本\orig_0.8_add.txt'
        expected_similarity = 75.12

        similarity = calculate_similarity(original_file, plagiarized_file)

        self.assertAlmostEqual(similarity, expected_similarity, places=2)

import time

def performance_test():
    # 记录开始时间
    start_time = time.time()

    # 执行主要功能代码
    main()

    # 记录结束时间
    end_time = time.time()

    # 计算程序执行时间
    execution_time = end_time - start_time

    # 打印执行时间
    print("程序执行时间: {:.2f}秒".format(execution_time))

# 在main函数调用之前添加性能测试模块
if __name__ == '__main__':
    try:
        performance_test()
    except Exception as e:
        print("系统正在升级中...........")

    #unittest.main()