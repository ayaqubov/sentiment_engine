#!/bin/bash
SHELL=/bin/bash
PATH=/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/opt/aws/bin

#/home/ayaqubov/.conda/envs/Python27/bin/python2 /home/ayaqubov/automation/engines/run_sentiment_engine.py

# synchronize each time since sometimes the result is lost

#sudo mount.cifs //ch01f15.corp.logitech.com/Sentiment /home/ayaqubov/engines/new_engine/DriverShare -o user=ayaqubov,vers=2.1,dir_mode=0777,file_mode=0777

/home/ayaqubov/.conda/envs/Python27/bin/python2 /home/ayaqubov/engines/new_engine/run_sentiment_eng_new.py
