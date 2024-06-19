#!/usr/bin/env python
# coding: utf-8

# In[1]:


import speech_recognition as sr
#語音辨識
r = sr.Recognizer()
mic = sr.Microphone()
a = 0
while True:
    with mic as source:
        #print("說出指令...")
        r.adjust_for_ambient_noise(source, duration = 0.8)  
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language="zh-TW")
            if text == '暫停':
                a = 1
                print("暫停")
            elif text == '播放':
                a = 2
                print("播放")
            elif text == '下一首':
                a = 3
                print("下一首")
            elif text == '上一首':
                a = 4
                print("上一首")
            else:
                print("無此指令")
            with open('writeSomething.txt', 'w') as f:
                f.write(str(a))
                f.close()
        except sr.UnknownValueError:
            print("請在說一次")

