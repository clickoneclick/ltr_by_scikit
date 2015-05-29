#!/usr/bin/python
#coding=utf-8

import sys
import pycurl
import time
import hashlib
import json

class Test:
  def __init__(self):
    self.contents = ''

  def body_callback(self, buf):
    self.contents = self.contents + buf

def get_puid_data(puid_list):
  uid='104'
  passwd='ganjituijian_!)@&_20141126'
  start_time = int(time.time())
  key=hashlib.md5('%s%s%s' % (uid, passwd, start_time)).hexdigest()

  url="http://tg.dns.ganji.com/api.php?c=FangQuery&a=getInfoByPuid&userid=%s&time=%s&key=%s" % (uid, start_time, key)
  #print url
  #puid_list='1238869334'
  #fields='puid,city,title'
  fields='puid,major_category,city,district_id,street_id,xiaoqu_id,price,area,huxing_shi,huxing_ting,zhuangxiu,image_count,ceng,ceng_total,chaoxiang,fang_xing,person,phone,title,latlng'
  para="puid=%s&fields=%s" % (puid_list, fields)
  #print para


  t = Test()
  c = pycurl.Curl()
  c.setopt(c.URL, url)
  c.setopt(c.POSTFIELDS, para)
  c.setopt(c.WRITEFUNCTION, t.body_callback)
  c.perform()
  end_time = time.time()
  duration = end_time - start_time
  #print c.getinfo(pycurl.HTTP_CODE), c.getinfo(pycurl.EFFECTIVE_URL)
  c.close()

  #print 'pycurl takes %s seconds to get %s ' % (duration, url)

  #print 'lenth of the content is %d' % len(t.contents)
  #print(t.contents)
  return t.contents

if __name__ == "__main__":
  data = get_puid_data(sys.argv[1])
  print data
  obj = json.loads(data)
  for item in obj['data']:
    try:
      for sub_key in obj['data'][item]:
        if sub_key == 'latlng':
          print obj['data'][item][sub_key]
    except:
      continue

