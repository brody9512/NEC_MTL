# test.py
import os
import shutil
import torch
import pandas as pd

import config
import utils

def main():

    args = config.get_args_test()
    

### again, we're not using CBAM and i think we dont need channelattention class either then? Also from MTL_test.py there are changes not yet made for what you gave guidance on so far since I'm only working on training part right now, but I just inputted for your reference on when putting it all together