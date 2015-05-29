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

def label(click_file, show_file):
  click_dict = {}
  for line in file(click_file, 'r'):
    puid_pair = line.split('\t')[1]
    click_dict[puid_pair] = (puid_pair in click_dict) and click_dict[puid_pair] + 1 or 1
  for line in file(show_file, 'r'):
    line = line.strip()
    puid_list = line.split('\t')[1].split(' ')
    click_list = [puid_list[0] + ':' + '0']
    for i in range(1, len(puid_list)):
      puid_pair = puid_list[0] + ' ' + puid_list[i]
      if puid_pair in click_dict:
        click_list.append(puid_list[i] + ':%d' % click_dict[puid_pair])
    if len(click_list) > 1:
      print '1\t' + ' '.join(click_list) + '\t' + line
    else:
      print '0\t' + ' '.join(click_list) + '\t' + line

def label_uuid(click_file, show_file):
  click_dict = {}
  for line in file(click_file, 'r'):
    try:
      line = line.strip()
      fields = line.split('\t')
      # key = uuid + src_puid + des_puid
      key = fields[3] + ' ' + extract_puid.extract_puid(fields[0]) + ' ' + fields[1]
      click_dict[key] = (key in click_dict) and click_dict[key] + 1 or 1
    except Exception as e:
      pass
  for line in file(show_file, 'r'):
    line = line.strip()
    fields = line.split('\t')
    puid_list = line.split('\t')[3].split('-')
    click_list = [puid_list[0] + ':' + '0']
    for i in range(1, len(puid_list)):
      key = fields[2] + ' ' + puid_list[0] + ' ' + puid_list[i]
      if key in click_dict:
        click_list.append(puid_list[i] + ':%d' % click_dict[key])
    if len(click_list) > 1:
      print '1\t' + '-'.join(click_list) + '\t' + line
    else:
      print '0\t' + '-'.join(click_list) + '\t' + line

if __name__ == "__main__":
  #label(sys.argv[1], sys.argv[2])
  label_uuid(sys.argv[1], sys.argv[2])

