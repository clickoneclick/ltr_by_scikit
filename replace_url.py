#!/usr/bin/python
#coding=utf-8

import string
import sys
import re
#url_pattern=re.compile(r"http://.*\.ganji\.com/fang1/(\(tuiguang-\)?\(\d+\)(x)?).htm")
url_pattern=re.compile(r"http://.*\.ganji\.com/fang1/(.*).htm.*")

def replace_url(url, puid):
  return url.replace(url_pattern.match(url).group(1), puid + 'x')

if __name__ == "__main__":
  for line in file(sys.argv[1], 'r'):
    field = line.split()
    print field[0], replace_url(field[0], field[1])

