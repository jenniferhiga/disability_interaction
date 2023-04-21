#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:12:11 2023

@author: jenniferhiga
"""

from flask import Flask
app = Flask(__name__)

@app.route('/')
def display():
    return "Looks like it works!"

if __name__=='__main__':
    app.run()