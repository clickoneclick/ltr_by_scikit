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

def print_click(filename):
  for line in file(filename, 'r'):
    try:
      line = line.strip()
      line = re.sub(r' +', ' ', line)
      fields = line.split('\t')
      fields[0] = re.sub(r'htm.*$', 'htm', fields[0])
      puid_list = []
      puid_list.append(extract_puid.extract_puid(fields[0]))
      #puid_list.append(extract_puid.extract_puid(fields[1]))
      puid_list.append(fields[1])
      data = get_curl.get_puid_data(','.join(puid_list))
      print fields[0] + '\t' + ' '.join(puid_list) + '\t' + data
      '''
      obj = json.loads(data)
      prop = sim(obj, puid_list[0], puid_list[1])
      print fields[0] + ' ' + replace_url.replace_url(fields[0], puid_list[1]) + ' ' + prop
      '''
    except Exception as e:
      pass
    break

def print_show(filename):
  for line in file(filename, 'r'):
    try:
      line = line.strip()
      line = re.sub(r' +', ' ', line)
      fields = line.split('\t')
      puid_list = []
      puid_list.append(extract_puid.extract_puid(fields[0]))
      #puid_list.append(extract_puid.extract_puid(fields[1]))
      fields[1] = re.sub(r't', '', fields[1])
      puid_list += fields[1].split('-')
      data = get_curl.get_puid_data(','.join(puid_list))
      del fields[1]
      print '\t'.join(fields) + '\t' + '-'.join(puid_list) + '\t' + data
      '''
      obj = json.loads(data)
      for i in range(1, len(puid_list)):
        try:
          prop = sim(obj, puid_list[0], puid_list[1])
          print fields[0] + ' ' + replace_url.replace_url(fields[0], puid_list[i]) + ' ' + prop
        except Exception as e:
          pass
      '''
    except Exception as e:
      pass

if __name__ == "__main__":
  #print_click(sys.argv[1])
  print_show(sys.argv[1])

