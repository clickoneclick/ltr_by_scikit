#!/usr/bin/python
#coding=utf-8

import sys
import pycurl
import time
import hashlib
import json
import re

import extract_puid
import get_curl
import get_distance
import replace_url
import feature_selection
import numpy as np
from numpy import asarray
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score

def gen_feature(obj, src_puid, des_puid):
  image_feature = 'invalid'
  des_image_feature = 'invalid'
  xiaoqu_feature = 'invalid'
  distance_feature = 'invalid'
  huxing_shi_feature = 'invalid'
  area_feature = 'invalid'
  area_ratio_feature = 'invalid'
  zhuangxiu_feature = 'invalid'
  price_feature = 'invalid'
  price_ratio_feature = 'invalid'
  if src_puid in obj['data'] and des_puid in obj['data']:
    try:
      bak_src_huxing = obj['data'][src_puid]['huxing_shi']
      bak_des_huxing = obj['data'][des_puid]['huxing_shi']
      bak_src_area = obj['data'][src_puid]['area']
      bak_des_area = obj['data'][des_puid]['area']
      if abs(float(obj['data'][src_puid]['price']) - float(obj['data'][des_puid]['price'])) / float(obj['data'][des_puid]['price']) < 2:
        # 价格差不多
        if int(obj['data'][src_puid]['huxing_shi']) > 2 and int(obj['data'][des_puid]['huxing_shi']) == 1:
          obj['data'][src_puid]['huxing_shi'] = 1
        if float(obj['data'][src_puid]['area']) > 2 * float(obj['data'][des_puid]['area']):
          obj['data'][src_puid]['area'] = float(obj['data'][src_puid]['area']) / float(obj['data'][src_puid]['huxing_shi'])
      if abs(float(obj['data'][des_puid]['price']) - float(obj['data'][src_puid]['price'])) / float(obj['data'][src_puid]['price']) < 2:
        # src puid is hezu
        if int(obj['data'][des_puid]['huxing_shi']) > 2 and int(obj['data'][src_puid]['huxing_shi']) == 1:
          obj['data'][des_puid]['huxing_shi'] = 1
        if float(obj['data'][des_puid]['area']) > 2 * float(obj['data'][src_puid]['area']):
          obj['data'][des_puid]['area'] = float(obj['data'][des_puid]['area']) / float(obj['data'][des_puid]['huxing_shi'])
    except Exception as e:
      pass
    try:
      image_feature  = int(obj['data'][src_puid]['image_count']) > 0 and '1' or '0'
      image_feature += '_'
      image_feature += int(obj['data'][des_puid]['image_count']) > 0 and '1' or '0'
      des_image_feature = int(obj['data'][des_puid]['image_count']) > 0 and '1' or '0'
    except Exception as e:
      pass
    try:
      xiaoqu_feature = str(int(obj['data'][src_puid]['xiaoqu_id'] == obj['data'][des_puid]['xiaoqu_id']))
    except Exception as e:
      pass
    try:
      location_0 = obj['data'][src_puid]['latlng'][1:].split(',')
      location_1 = obj['data'][des_puid]['latlng'][1:].split(',')
      delta = 10 * get_distance.distance(float(location_0[0]), float(location_0[1]), float(location_1[0]), float(location_1[1]))
      if 0 == delta:
        distance_feature = '0'
      elif 0 < delta and delta <= 10:
        distance_feature = '1'
      elif 10 < delta and delta <= 30:
        distance_feature = '2'
      elif 30 < delta and delta <= 50:
        distance_feature = '3'
      elif 50 < delta:
        distance_feature = '4'
      elif -10 <= delta and delta < 0:
        distance_feature = '-1'
      elif -30 <= delta and delta < -10:
        distance_feature = '-2'
      elif -50 <= delta and delta < -30:
        distance_feature = '-3'
      elif delta < 50:
        distance_feature = '-4'
    except Exception as e:
      pass
    try:
      src_value = int(obj['data'][src_puid]['huxing_shi'])
      if src_value > 3: src_value = 4
      des_value = int(obj['data'][des_puid]['huxing_shi'])
      if des_value > 3: des_value = 4
      if src_value != des_value:
        src_value = 4
        des_value = 4
      huxing_shi_feature  = '%d_%d' % (src_value, des_value)
    except Exception as e:
      pass
    try:
      src_value = int(obj['data'][src_puid]['zhuangxiu'])
      if src_value > 4: src_value = 5
      des_value = int(obj['data'][des_puid]['zhuangxiu'])
      if des_value > 4: des_value = 5
      if src_value == 0:
        if des_value != 0 and des_value != 2:
          des_value = 5
      elif src_value == 2:
        if des_value != 2 and des_value != 3:
          des_value = 5
      elif src_value == 3:
        if des_value != 2 and des_value != 3:
          des_value = 5
      elif src_value == 4:
        if des_value != 2 and des_value != 3:
          des_value = 5
      elif src_value == 5 or src_value == 1:
        src_value = 5
        des_value = 5
      zhuangxiu_feature  = '%d_%d' % (src_value, des_value)
    except Exception as e:
      pass
    try:
      delta = int(10 * (float(obj['data'][des_puid]['area']) - float(obj['data'][src_puid]['area'])) / float(obj['data'][src_puid]['area']))
      if 0 < delta and delta <= 2:
        area_ratio_feature = '1'
      elif 2 < delta and delta <= 4:
        area_ratio_feature = '2'
      elif 4 < delta and delta <= 6:
        area_ratio_feature = '3'
      elif 6 < delta and delta <= 12:
        area_ratio_feature = '4'
      elif 12 < delta:
        area_ratio_feature = '5'
      elif -2 <= delta and delta < 0:
        area_ratio_feature = '-1'
      elif -4 <= delta and delta < -2:
        area_ratio_feature = '-2'
      elif -6 <= delta and delta < -4:
        area_ratio_feature = '-3'
      elif delta < -6:
        area_ratio_feature = '-4'
    except Exception as e:
      pass
    try:
      delta = int(float(obj['data'][des_puid]['area']) - float(obj['data'][src_puid]['area']))
      if 0 == delta:
        area_feature = '0'
      elif 0 < delta and delta <= 10:
        area_feature = '1'
      elif 10 < delta and delta <= 20:
        area_feature = '2'
      elif 20 < delta and delta <= 30:
        area_feature = '3'
      elif 30 < delta:
        area_feature = '4'
      elif -10 <= delta and delta < 0:
        area_feature = '-1'
      elif -20 <= delta and delta < -10:
        area_feature = '-2'
      elif -30 <= delta and delta < -20:
        area_feature = '-3'
      elif delta < -30:
        area_feature = '-4'
    except Exception as e:
      pass
    try:
      delta = int(obj['data'][des_puid]['price']) - int(obj['data'][src_puid]['price'])
      if 0 == delta:
        price_feature = '0'
      elif 0 < delta and delta <= 100:
        price_feature = '1'
      elif 100 < delta and delta <= 300:
        price_feature = '2'
      elif 300 < delta:
        price_feature = '3'
      elif -100 <= delta and delta < 0:
        price_feature = '-1'
      elif -300 <= delta and delta < -100:
        price_feature = '-2'
      elif delta < -300:
        price_feature = '-3'
    except Exception as e:
      pass
    try:
      delta = int(10 * (float(obj['data'][des_puid]['price']) - float(obj['data'][src_puid]['price'])) / float(obj['data'][src_puid]['price']))
      if 0 < delta and delta <= 1:
        price_ratio_feature = '1'
      elif 1 < delta and delta <= 2:
        price_ratio_feature = '2'
      elif 2 < delta:
        price_ratio_feature = '3'
      elif -1 <= delta and delta < 0:
        price_ratio_feature = '-1'
      elif -2 <= delta and delta < -1:
        price_ratio_feature = '-2'
      elif delta < -2:
        price_ratio_feature = '-3'
    except Exception as e:
      pass
    try:
      obj['data'][src_puid]['huxing_shi'] = bak_src_huxing 
      obj['data'][des_puid]['huxing_shi'] = bak_des_huxing 
      obj['data'][src_puid]['area'] = bak_src_area 
      obj['data'][des_puid]['area'] = bak_des_area 
    except Exception as e:
      pass
  #print fields[0], replace_url.replace_url(fields[0], fields[1]), same_xiaoqu, dis, price_difference, have_image, both_image
  result = {}
  result['image_feature'] = image_feature
  result['des_image_feature'] = des_image_feature
  result['xiaoqu_feature'] = xiaoqu_feature
  result['distance_feature'] = distance_feature
  result['huxing_shi_feature'] = huxing_shi_feature
  result['area_feature'] = area_feature
  result['area_ratio_feature'] = area_ratio_feature
  result['zhuangxiu_feature'] = zhuangxiu_feature
  result['price_feature'] = price_feature
  result['price_ratio_feature'] = price_ratio_feature
  single_feature = {}
  for item in result:
    if result[item] == 'invalid':
      continue
    feature_name = item + ':' + result[item]
    single_feature[feature_name] = 1
  return single_feature


def combine_feature(single_feature):
  compound_feature = {}
  for key in single_feature:
    compound_feature[key] = 1
  fea_list = single_feature.items()
  for i in range(0, len(fea_list)):
    for j in range(i + 1, len(fea_list)):
      feature_name = fea_list[i][0] + '@' + fea_list[j][0]
      compound_feature[feature_name] = 1
  return compound_feature

def is_same_sample(sample_feature, new_sample):
  feature_name = ['des_image_feature', 'xiaoqu_feature', 'huxing_shi_feature'\
      'area_feature', 'zhuangxiu_feature', 'price_feature']
  dis_num = 0
  for item in sample_feature.items():
    try:
      if item[0].find('image_feature') == 0 or item[0].find('distance_feature') == 0 or item[0].find('area_ratio_feature') == 0 or item[0].find('price_ratio_feature') == 0:
        continue
      if item[0] not in new_sample:
        dis_num += 1
    except Exception as e:
      pass
  if dis_num >= 2:
    return False
  return True

def parse_sample_in_block_for_test(url, obj, click_dict, puid_list, total_sample):
  # step 1. get values of all the shown puid for each feature
  valid_sample = []
  pos_sample = []
  pos_pos = 0
  for i in range(1, len(puid_list)):
    sample_feature = {}
    try:
      single_feature = gen_feature(obj, puid_list[0], puid_list[i])
      if len(single_feature) == 0:
        continue
      if puid_list[i] in click_dict:
        sample_feature = combine_feature(single_feature)
        sample_feature['label:1'] = 1
      else:
        sample_feature = combine_feature(single_feature)
        sample_feature['label:0'] = 1
      sample_feature['src_puid'] = puid_list[0]
      sample_feature['des_puid'] = puid_list[i]
      sample_feature['url'] = url
      valid_sample.append(sample_feature)
    except Exception as e:
      pass
  if len(valid_sample) > 0:
    total_sample.append(valid_sample)
  return len(valid_sample)


def parse_sample_in_block(url, obj, click_dict, puid_list, total_sample):
  # step 1. get values of all the shown puid for each feature
  valid_sample = []
  pos_sample = []
  pos_pos = 0
  for i in range(1, len(puid_list)):
    sample_feature = {}
    try:
      single_feature = gen_feature(obj, puid_list[0], puid_list[i])
      if len(single_feature) == 0:
        continue
      if puid_list[i] in click_dict:
        pos_sample.append(single_feature)
        pos_pos = i
      sample_feature = combine_feature(single_feature)
    except Exception as e:
      pass

  for i in range(1, len(puid_list)):
    sample_feature = {}
    try:
      single_feature = gen_feature(obj, puid_list[0], puid_list[i])
      if len(single_feature) == 0:
        continue
      if puid_list[i] in click_dict:
        sample_feature = combine_feature(single_feature)
        sample_feature['label:1'] = 1
      else:
        # 负样本去重
        repeat_flag = 0
        for item in pos_sample:
          #if sample_feature == item:
          if is_same_sample(single_feature, item):
            repeat_flag = 1
            break
        if repeat_flag == 1:
          continue
        # 负样本内部也进行去重
        #pos_sample.append(sample_feature.copy())
        sample_feature = combine_feature(single_feature)
        sample_feature['label:0'] = 1
      sample_feature['src_puid'] = puid_list[0]
      sample_feature['des_puid'] = puid_list[i]
      sample_feature['url'] = url
      valid_sample.append(sample_feature)
    except Exception as e:
      pass
  has_neg = 0
  has_pos = 0
  for sample in valid_sample:
    if 'label:0' in sample:
      has_neg += 1
    if 'label:1' in sample:
      has_pos += 1
  if has_neg > 0 and has_pos > 0:
    for sample in valid_sample:
      total_sample.append(sample)
    #print len(valid_sample), has_pos, has_neg
  return len(valid_sample)


def stat_result(total_sample):
  pos_sample_num = 0
  neg_sample_num = 0
  pos_total_stat = {}
  neg_total_stat = {}
  for sample in total_sample:
    if 'label:1' in sample:
      pos_sample_num += 1
      for key in sample:
        pos_total_stat[key] = key in pos_total_stat and pos_total_stat[key] + 1 or 1
    elif 'label:0' in sample:
      neg_sample_num += 1
      for key in sample:
        neg_total_stat[key] = key in neg_total_stat and neg_total_stat[key] + 1 or 1
  print 'pos_sample_num:%d' % pos_sample_num
  print 'neg_sample_num:%d' % neg_sample_num

  #print 'sample feature distribution'
  fea_stat = {'pos':{}, 'neg':{}}
  puid_stat = {'pos':{}, 'neg':{}}
  for sample in total_sample:
    tmp = sample.copy()
    del tmp['src_puid']
    del tmp['des_puid']
    del tmp['url']
    if 'label:1' in sample:
      puid_pair = sample['src_puid'] + ':' + sample['des_puid']
      del tmp['label:1']
      key = '#'.join(sorted(tmp.keys()))
      fea_stat['pos'][key] = key in fea_stat['pos'] and fea_stat['pos'][key] + 1 or 1
      '''
      if key in puid_stat['pos']:
        puid_stat['pos'][key].append(puid_pair)
      else:
        puid_stat['pos'][key] = [sample['url'], puid_pair]
      '''
    elif 'label:0' in sample:
      puid_pair = sample['src_puid'] + ':' + sample['des_puid']
      del tmp['label:0']
      key = '#'.join(sorted(tmp.keys()))
      fea_stat['neg'][key] = key in fea_stat['neg'] and fea_stat['neg'][key] + 1 or 1
      '''
      if key in puid_stat['pos']:
        puid_stat['pos'][key].append(puid_pair)
      else:
        puid_stat['pos'][key] = [sample['url'], puid_pair]
      '''
  feature_filter = {}
  for item in sorted(fea_stat['pos'].items(), key = lambda a:a[0]):
    feature_name = item[0]
    feature_in_pos = item[1]
    feature_in_neg = feature_name in fea_stat['neg'] and fea_stat['neg'][feature_name] or 0
    '''
    if feature_name not in puid_stat['neg']:
      puid_stat['neg'][feature_name] = ['']
    '''
    # 只选择点击数大于4的样本
    if feature_in_pos > 4:
      # 分析特征数据
      #print '%s\t%d\t%d\t%s\t%s' % (feature_name, feature_in_pos, feature_in_neg, '-'.join(puid_stat['pos'][feature_name]), '-'.join(puid_stat['neg'][feature_name]))
      for fea in feature_name.split('#'):
        num = feature_in_pos + feature_in_neg
        feature_filter[fea] = fea in feature_filter and feature_filter[fea] + num or num
  '''
  for item in sorted(feature_filter.items(), key = lambda a:a[1], reverse = True):
    if item[1] > 4:
      # 输出特征及频次
      print item[0], item[1]
      pass

  '''
  print 'single feature distribution'
  for item in sorted(pos_total_stat.items(), key = lambda a:a[0]):
    feature_name = item[0]
    feature_in_pos = item[1]
    feature_in_neg = feature_name in neg_total_stat and neg_total_stat[feature_name] or 0
    # feature_name feature_in_pos feature_in_neg
    click_ratio = float(feature_in_pos) / float(feature_in_pos + feature_in_neg)
    chi_value = feature_selection.chisq_test(pos_sample_num, neg_sample_num, feature_in_pos, feature_in_neg)
    ig_value = feature_selection.information_gain(pos_sample_num, neg_sample_num, feature_in_pos, feature_in_neg)
    right_ratio = float(float(feature_in_pos) / pos_sample_num)
    fault_ratio = float(float(feature_in_neg) / neg_sample_num)
    print '%s\t%d\t%d\t%f\t%f\t%f\t%f\t%f' % (feature_name, feature_in_pos, feature_in_neg, click_ratio, chi_value, ig_value, right_ratio, fault_ratio)

#A helper method for pretty-printing linear models
def pretty_print_linear(feature_dict, coefs, names = None, sort = False):
  if names == None:
    names = [item[0] for item in sorted(feature_dict.items(), key = lambda a:a[1])]
  lst = zip(coefs[0], names)
  if sort:
    lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
  print "\n".join("%s : %s" % (name, round(coef, 3)) for coef, name in lst)

def load_feature_dict_single_feature(feature_dict, line_num):
  feature_num = 0
  count = 0
  for line in file('mid/feature_dict', 'r'):
    if count != line_num:
      count += 1
      continue
    feature = line.split(' ')[0]
    print feature,
    if feature not in feature_dict:
      feature_dict.setdefault(feature, feature_num)
      feature_num += 1
    count += 1
  #print sorted(feature_dict.items(), key = lambda a:a[1])
  pass

def load_feature_dict(feature_dict):
  feature_num = 0
  count = 0
  for line in file('mid/feature_dict', 'r'):
    feature = line.split(' ')[0]
    if feature not in feature_dict:
      feature_dict.setdefault(feature, feature_num)
      feature_num += 1
    count += 1
  #print sorted(feature_dict.items(), key = lambda a:a[1])

def train_lr_single_feature(total_sample):
  for line_num in range(0, 10):
    feature_dict = {}
    load_feature_dict_single_feature(feature_dict, line_num)
    fea_len = len(feature_dict)
    x = []
    y = []
    for sample in total_sample:
      arr = [0] * (fea_len)
      #arr[fea_len] = 1
      for feature in sample:
        if 'label:1' == feature:
          y.append(1)
        elif 'label:0' == feature:
          y.append(0)
        else:
          if feature in feature_dict:
            arr[feature_dict[feature]] = 1
      x.append(arr)
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_validation import cross_val_score
    x = asarray(x)
    y = asarray(y)
    clf2 = LogisticRegression()
    score = cross_val_score(clf2, x, y, scoring = 'roc_auc', cv = 2)
    print line_num, score
    '''
    clf2 = LogisticRegression().fit(x, y)
    print "LogisticRegression model:"
    pretty_print_linear(feature_dict, clf2.coef_, sort=True)
    '''

def train_lr_dcg(train_sample, test_sample):
  feature_dict = {}
  load_feature_dict(feature_dict)
  fea_len = len(feature_dict)
  x = []
  y = []
  for sample in train_sample:
    arr = [0] * (fea_len)
    #arr[fea_len] = 1
    for feature in sample:
      if 'label:1' == feature:
        y.append(1)
      elif 'label:0' == feature:
        y.append(0)
      else:
        if feature in feature_dict:
          arr[feature_dict[feature]] = 1
    x.append(arr)
  x = asarray(x)
  y = asarray(y)
  '''
  clf2 = LogisticRegression()
  score = cross_val_score(clf2, x, y, scoring = 'roc_auc', cv = 3)
  print score
  '''
  clf2 = LogisticRegression().fit(x, y)
  block_num = 0
  true_pos = {}
  predict_pos = {}
  for i in range(10):
    true_pos[i] = 0
    predict_pos[i] = 0
  print len(train_sample)
  print len(test_sample)
  for block in test_sample:
    block_num += 1
    x_test = []
    y_test = []
    for sample in block:
      arr = [0] * (fea_len)
      #arr[fea_len] = 1
      for feature in sample:
        if 'label:1' == feature:
          y_test.append(1)
        elif 'label:0' == feature:
          y_test.append(0)
        else:
          if feature in feature_dict:
            arr[feature_dict[feature]] = 1
      x_test.append(arr)
    x_test = asarray(x_test)
    y_test = asarray(y_test)
    y_score = clf2.predict_proba(x_test)[:, 1]
    y_predict = []
    for i in range(len(y_score)):
      y_predict.append((i, y_score[i]))
    y_predict = sorted(y_predict, key = lambda a:a[1], reverse = True)
    for i in range(len(y_test)):
      true_pos[i] += y_test[i]
      if y_test[i] == 1:
        for j in range(len(y_predict)):
          if i == y_predict[j][0]:
            predict_pos[j] += 1
  print true_pos
  print predict_pos


def train_lr_auc(train_sample, test_sample):
  feature_dict = {}
  load_feature_dict(feature_dict)
  fea_len = len(feature_dict)
  x = []
  y = []
  for sample in train_sample:
    arr = [0] * (fea_len)
    #arr[fea_len] = 1
    for feature in sample:
      if 'label:1' == feature:
        y.append(1)
      elif 'label:0' == feature:
        y.append(0)
      else:
        if feature in feature_dict:
          arr[feature_dict[feature]] = 1
    x.append(arr)
  x = asarray(x)
  y = asarray(y)
  '''
  clf2 = LogisticRegression()
  score = cross_val_score(clf2, x, y, scoring = 'roc_auc', cv = 3)
  print score
  '''
  clf2 = LogisticRegression().fit(x, y)
  x_test = []
  y_test = []
  for block in test_sample:
    for sample in block:
      arr = [0] * (fea_len)
      #arr[fea_len] = 1
      for feature in sample:
        if 'label:1' == feature:
          y_test.append(1)
        elif 'label:0' == feature:
          y_test.append(0)
        else:
          if feature in feature_dict:
            arr[feature_dict[feature]] = 1
      x_test.append(arr)
  x_test = asarray(x_test)
  y_test = asarray(y_test)

  y_score = clf2.predict_proba(x_test)[:, 1]
  print roc_auc_score(y_test, y_score)
  #print "LogisticRegression model:"
  #pretty_print_linear(feature_dict, clf2.coef_, sort=True)

def load_sample_test(testfile, test_sample):
  total_stat = [{}, {}]
  sample_num = [0, 0]
  click_num_stat = {}
  click_pos_stat = {}
  total_sample_num = 0
  valid_sample_num = 0
  for line in file(testfile, 'r'):
    try:
      line = line.strip()
      fields = line.split('\t')
      # filter the negtive sample
      #if fields[0] == '0':
        #continue
      click_puid_dict = {}
      map(lambda m:click_puid_dict.setdefault(m.split(':')[0], 1), fields[1].split('-')[1:])
      show_puid_list = fields[5].split('-')
      obj = json.loads(fields[6])
      # stat the click num distribution, click pos distribution
      click_num = 0
      for pos in range(1, len(show_puid_list)):
        if show_puid_list[pos] in click_puid_dict:
          click_num += 1
          key = 'click_pos*%d' % pos
          click_pos_stat[key] = key in click_pos_stat and click_pos_stat[key] + 1 or 1
      key = 'click_num*%d' % click_num
      click_num_stat[key] = key in click_num_stat and click_num_stat[key] + 1 or 1
      total_sample_num += len(show_puid_list) - 1
      # stat the feature num distribution
      valid_sample_num += parse_sample_in_block_for_test(fields[2], obj, click_puid_dict, show_puid_list, test_sample)
    except Exception as e:
      pass
    #break

def load_sample_train(filename, total_sample):
  total_stat = [{}, {}]
  sample_num = [0, 0]
  click_num_stat = {}
  click_pos_stat = {}
  total_sample_num = 0
  valid_sample_num = 0
  for line in file(filename, 'r'):
    try:
      line = line.strip()
      fields = line.split('\t')
      # filter the negtive sample
      if fields[0] == '0':
        continue
      click_puid_dict = {}
      map(lambda m:click_puid_dict.setdefault(m.split(':')[0], 1), fields[1].split('-')[1:])
      show_puid_list = fields[5].split('-')
      obj = json.loads(fields[6])
      # stat the click num distribution, click pos distribution
      click_num = 0
      for pos in range(1, len(show_puid_list)):
        if show_puid_list[pos] in click_puid_dict:
          click_num += 1
          key = 'click_pos*%d' % pos
          click_pos_stat[key] = key in click_pos_stat and click_pos_stat[key] + 1 or 1
      key = 'click_num*%d' % click_num
      click_num_stat[key] = key in click_num_stat and click_num_stat[key] + 1 or 1
      total_sample_num += len(show_puid_list) - 1
      # stat the feature num distribution
      valid_sample_num += parse_sample_in_block(fields[2], obj, click_puid_dict, show_puid_list, total_sample)
    except Exception as e:
      pass
    #break

  '''
  print 'total_sample_num:%d' % total_sample_num
  print 'valid_sample_num:%d' % len(total_sample)
  print 'click num distribution'
  for item in click_num_stat:
    print '%s\t%d' % (item, click_num_stat[item])
  print 'click pos distribution'
  for item in sorted(click_pos_stat.items(), key = lambda a:a[0]):
    print '%s\t%d' % (item[0], item[1])
  '''


def parse_data_to_sample(trainfile, testfile):
  train_sample = []
  test_sample = []
  load_sample_train(trainfile, train_sample)
  load_sample_test(testfile, test_sample)
  #stat_result(train_sample)
  #train_lr_single_feature(train_sample)
  train_lr_auc(train_sample, test_sample)
  #train_lr_dcg(train_sample, test_sample)


if __name__ == "__main__":
  parse_data_to_sample(sys.argv[1], sys.argv[2])

