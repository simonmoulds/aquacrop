#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import os
# import numpy as np
# import netCDF4 as nc
import sqlite3
# from importlib_resources import path


def db_connect(db_path):
    con = sqlite3.connect(db_path)
    return con


def get_parameter_id(con, parameter_name):
    with con:
        query = "SELECT id FROM parameter_values WHERE name = '" + parameter_name + "'"
        cur = con.cursor()
        cur.execute(query)
        val = cur.fetchone()[0]
    return val


def read_crop_parameter_from_sqlite(con, crop_id, parameter_name):
    parameter_id = get_parameter_id(con, parameter_name)
    with con:
        query = "SELECT value,min,max FROM parameter_values WHERE crop_id = " + \
            str(crop_id) + " and parameter_id = " + str(parameter_id)
        cur = con.cursor()
        cur.execute(query)
        val = cur.fetchone()
    return val


def read_parameter_from_sqlite(con, parameter_name):
    parameter_id = get_parameter_id(con, parameter_name)
    with con:
        query = "SELECT value,min,max FROM parameter_values WHERE parameter_id = " + \
            str(parameter_id)
        cur = con.cursor()
        cur.execute(query)
        val = cur.fetchone()
    return val
