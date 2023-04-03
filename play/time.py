from threading import Thread
from time import sleep
def work():
    sleep(1)
    print('3')
    sleep(1)
    print('2')
    sleep(1)
    print('1')

t = Thread(target=work)
t.start()

while t.is_alive():
    sleep(0.1)
    print('waiting')

t.join()

print('done w whole thing')
