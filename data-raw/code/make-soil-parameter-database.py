#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sqlite3
import pandas as pd

# DEFAULT_PATH = os.path.join(os.path.dirname(__file__), 'database.sqlite3')
DEFAULT_PATH = 'soil_parameter_database.sqlite3'
os.system('rm soil_parameter_database.sqlite3')

def db_connect(db_path=DEFAULT_PATH):
    con = sqlite3.connect(db_path)
    return con

# def get_crops():
#     with con:
#         cur.execute("SELECT * FROM crops")
#         print((cur.fetchall()))

# def get_parameters():
#     with con:
#         cur.execute("SELECT * FROM parameters")
#         print((cur.fetchall()))

# def get_parameter_value(crop_id, parameter_id):
#     with con:
#         cur.execute("SELECT value,min,max FROM parameter_values WHERE crop_id = " + str(crop_id) + " and parameter_id = " + str(parameter_id))
#         print((cur.fetchone()))
#     # return val

def run():
    con = db_connect()
    cur = con.cursor()
    # crops_sql = """
    # CREATE TABLE crops (
    # id integer PRIMARY KEY,
    # name text NOT NULL)"""
    # cur.execute(crops_sql)
    parameters_sql = """
    CREATE TABLE parameters (
    id integer PRIMARY KEY,
    name text,
    description text,
    section text,
    subsection text,
    type text,
    units text)"""
    cur.execute(parameters_sql)
    parameter_values_sql = """
    CREATE TABLE parameter_values (
    id integer,
    name real,
    value real,
    min real,
    max real,
    parameter_id integer,
    FOREIGN KEY (parameter_id) REFERENCES parameters(id))"""
    cur.execute(parameter_values_sql)

    parameters_df = pd.read_csv('../data/soil_parameters.csv', encoding='utf-8')
    parameters_df.to_sql('parameters', con, if_exists='replace', index=False)
    vals_df = pd.read_csv(os.path.join('../data','soil_parameter_values.csv'), encoding='utf-8')
    vals_df.to_sql('parameter_values', con, if_exists='append', index=False)
    os.system("cp soil_parameter_database.sqlite3 ../../aquacrop/io/data")

if __name__ == "__main__":
    run()
