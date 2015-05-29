#!/usr/bin/python
#coding=utf-8

import sys
import pycurl
import time
import hashlib
import json
import re

import extract_puid

def add_url(raw_file, data_file):
  url_file = file(raw_file, 'r')
  for line in file(data_file, 'r'):
    try:
      line = line.strip()
      line = re.sub(r' +', ' ', line)
      fields = line.split('\t')
      puid_list = fields[0].split(' ')
      url = ''
      while True:
        url = url_file.readline().split('\t')[0]
        url = re.sub(r'htm.*$', 'htm', url)
        if extract_puid.extract_puid(url) == puid_list[0]:
          print url + '\t' + line
          break
    except Exception as e:
      pass

if __name__ == "__main__":
  add_url(sys.argv[1], sys.argv[2])

