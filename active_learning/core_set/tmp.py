# flush

import time
import sys


if __name__=="__main__":
    for i in range(5):
        print(i, flush=True)
        # time.sleep(1)
        if i == 4:
            sys.stdout.flush()

    # for i in range(5):
    #     print(i)
    #     sys.stdout.flush()
    #     time.sleep(1)
    #
    #
    # for i in range(10):
    #     print(i)
    #     if i == 5:
    #         print("Flushing buffer")
    #         sys.stdout.flush()    # todo: 이거 뭐 한걸까 ... ?
    #     time.sleep(1)
    #
    #
    # for i in range(10):
    #     print(i)
    #     if i == 5:
    #         print("Flushing buffer")
    #         sys.stdout.flush()