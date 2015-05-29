#!/usr/bin/python
#coding=utf-8

import sys
import pycurl
import time
import hashlib
import json
import re

import extract_puid

def add_url(raw_file):
  for line in file(raw_file, 'r'):
    try:
      line = line.strip()
      fields = line.split('\t')
      fields[0] = re.sub(r'htm.*$', 'htm', fields[0])
      print '\t'.join(fields)
    except Exception as e:
      pass

if __name__ == "__main__":
  add_url(sys.argv[1])

