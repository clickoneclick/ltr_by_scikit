#!/usr/bin/python
#coding=utf-8

import string
import sys
import re
url_pattern=re.compile(r"http://.*\.ganji\.com/fang1/(tuiguang-)?(\d+)x?\.htm")

def extract_puid(url):
  return url_pattern.match(url).group(2)

if __name__ == "__main__":
  url = "http://xm.ganji.com/fang5/tuiguang-73714750.htm"
  print extract_puid(url)
  url = "http://shaoxing.ganji.com/fang1/1423274539x.htm"
  print extract_puid(url)

