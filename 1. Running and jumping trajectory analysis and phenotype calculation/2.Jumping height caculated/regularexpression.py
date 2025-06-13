#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 16:26:30 2021

@author: hjh
"""
import re

class Re0:
    def name(self, allname):
        self.allname = allname
        pattern = re.compile(r'''
                           ([SMT]\d{1,3}_ 
                           .{1,10}_
                           \d{1,10}_
                           .{1,10}_
                           ([123]|11))|
                           ([SMT]\d{1,3}_.{1,10}_[123])
                           ''', re.X)
        result = pattern.search(self.allname)
        if result == None:
            return None
        return result.group()


if __name__ == '__main__':
    r = Re0()
    path = '/home/hjh/DLC1/output/jump/XH_jump/XH_jump_7.5/S31_022_24_066_2DLC_resnet50_XH_jump_10M(S)Apr13shuffle1_50000_bx_filtered.csv'
                                                           
    x = r.name(path)    
    print(x)