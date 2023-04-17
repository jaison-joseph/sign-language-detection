from threading import Thread, Lock
from time import sleep

lock = Lock()
start_lock = Lock()
doWork = True

def work():
    global doWork, startWork, lock, start_lock
    start_lock.acquire()
    # sleep(1)
    while True:
        lock.acquire()
        if not doWork:
            return
        for i in range(10):
            print(i)
            sleep(0.1)
        lock.release()
        sleep(0.5)

t = Thread(target = work)
start_lock.acquire()
t.start()
start_lock.release()

while True:
    lock.acquire()
    x = input('1 to start, anything else to quit')
    if x == '1':
        lock.release()
        sleep(0.5)
    else:
        lock.release()
        doWork = False
        break

t.join()


# t.start()

# t.join()
# print('from main: t is no longer alive')
# print(t.is_alive())

# t.start()
# t.join()
# print('again, t is done')
# print(t.is_alive())
