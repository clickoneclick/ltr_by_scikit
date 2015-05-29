#!/usr/bin/python
#coding=utf-8

import string
import sys
import re
import scipy
import math
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import pearsonr
np.random.seed(0)

#url_pattern=re.compile(r"http://.*\.ganji\.com/fang1/(\(tuiguang-\)?\(\d+\)(x)?).htm")
url_pattern=re.compile(r"http://.*\.ganji\.com/fang1/(.*).htm.*")

#距离相关系数 (Distance correlation)

#最大信息系数 maximal information coefficient (MIC)


#卡方检验
def chisq_test(pos_sample_num, neg_sample_num, feature_in_pos, feature_in_neg):
  n = float(pos_sample_num + neg_sample_num)
  a = float(feature_in_pos)
  b = float(feature_in_neg)
  c = float(pos_sample_num - a)
  d = float(neg_sample_num - c)
  value = n*((a*d - b*c)**2)/((a+b)*(c+d)*(a+c)*(b+d))
  return value

#pearson相关系数
def pearson_test(x, y):
  return pearsonr(x, y)

#information gain信息增益
def information_gain(pos_sample_num, neg_sample_num, feature_in_pos, feature_in_neg):
  infoGain = 0.0
  try:
    N1 = float(pos_sample_num) > 1.0 and float(pos_sample_num) or 1.0
    N2 = float(neg_sample_num) > 1.0 and float(neg_sample_num) or 1.0
    entropy = -((N1/(N1 + N2)) * math.log(N1/(N1 + N2)) + (N2/(N1 + N2)) * math.log(N2/(N1 + N2)))
    A = float(feature_in_pos) > 1.0 and float(feature_in_pos) or 1.0
    B = float(feature_in_neg) > 1.0 and float(feature_in_neg) or 1.0
    C = N1 - A
    D = N2 - B
    infoGain = entropy + (A + B)/(N1 + N2) * (A/(A + B) * math.log(A/(A + B)) + B/(A + B) * math.log(B/(A + B))) + (C + D)/(N1 + N2) * (C/(C + D) * math.log(C/(C + D)) + D/(C + D) * math.log(D/(C + D)))
  except Exception as e:
    pass
  return infoGain

#线性模型和正则化
#A helper method for pretty-printing linear models
def pretty_print_linear(coefs, names = None, sort = False):
  if names == None:
    names = ["X%s" % x for x in range(len(coefs))]
  lst = zip(coefs, names)
  if sort:
    lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
  return " + ".join("%s * %s" % (round(coef, 3), name) for coef, name in lst)

def linear_regression_test(x, y):
  lr = LinearRegression()
  lr.fit(x, y)
  return pretty_print_linear(lr.coef_)

if __name__ == "__main__":
  pos_sample_num = 20371
  neg_sample_num = 68095
  feature_in_pos = 4596
  feature_in_neg = 15483
  print chisq_test(pos_sample_num, neg_sample_num, feature_in_pos, feature_in_neg)
  print information_gain(pos_sample_num, neg_sample_num, feature_in_pos, feature_in_neg)

  size = 300
  x = np.random.normal(0, 1, size)
  y = x + np.random.normal(0, 1, size)
  print "Lower noise", pearson_test(x, y)
  y = x + np.random.normal(0, 10, size)
  print "Higher noise", pearson_test(x, y)

  size = 300
  x = np.random.normal(0, 1, (size, 3))
  y = np.random.normal(0, 1, size)
  print "Linear model:", linear_regression_test(x, y)
 
