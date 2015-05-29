import math
import get_curl

def distance(lat1,lng1,lat2,lng2):
  radlat1 = math.radians(lat1)
  radlat2 = math.radians(lat2)
  a = radlat1 - radlat2
  b = math.radians(lng1) - math.radians(lng2)
  s = 2*math.asin(math.sqrt(math.pow(math.sin(a/2),2)+math.cos(radlat1)*math.cos(radlat2)*math.pow(math.sin(b/2),2)))
  earth_radius = 6378.137
  s=s*earth_radius
  return abs(s)

if __name__ == "__main__":
  data = (113.274894, 23.164966, 113.280873, 23.169371)
  print distance(data[0], data[1], data[2], data[3])
  #print distance(116.383411734,40.1220402987,116.425704,40.112674)

