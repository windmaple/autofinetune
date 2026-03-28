import time
#from uiautomator import Device
#d = Device('3e499882')
from uiautomator import device as d
for i in range(10000):
  time.sleep(10)
  d.swipe(180, 650, 180, 200)

