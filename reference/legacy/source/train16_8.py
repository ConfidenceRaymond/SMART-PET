import time
import torch
from param import Opts
from main16_8 import Net
from utils_smart import misc
torch.autograd.set_detect_anomaly(True)
opt = Opts()
misc.seed_everything(2023)


def main(action='train'):
    SynModel = Net()    

    if action=='train':
        SynModel.train()
            
    else:
        SynModel.test()
        

if __name__ == '__main__':
    start_time = time.time()
    main(action='test')
    print("--- %s seconds ---" % (time.time() - start_time))
