import os
import sqlite3
import sqlalchemy
import pandas
import numpy as np
from typing import Callable

DBASE = 'dataset/corpus_working_copy.sqlite3'
PKL_CACHE_FOLDER = "pkl_cache"


def get_pandas_from_table(table_name):
    con = sqlalchemy.create_engine("sqlite:///" + DBASE).connect()
    pandas_frame = pandas.read_sql_table(table_name, con)
    con.close()
    return pandas_frame


def generate_pkl_cached(pkl_cache_fname: str, function: Callable[[any], pandas.DataFrame], *args, **kwargs):
    if pkl_cache_fname is not None:
        try:
            if not os.path.exists(PKL_CACHE_FOLDER):
                os.makedirs(PKL_CACHE_FOLDER)
            return pandas.read_pickle(PKL_CACHE_FOLDER + "/" + pkl_cache_fname)
        except (FileNotFoundError, IOError):
            frame = function(*args, **kwargs)
            frame.to_pickle(PKL_CACHE_FOLDER + "/" + pkl_cache_fname)
            return frame
    else:
        return function(*args, **kwargs)


def query_to_data_frame(sql: str, pkl_cache_fname):
    def load_data():
        con = sqlite3.connect(DBASE)
        curs = con.cursor()
        curs.execute(sql)
        frame = pandas.DataFrame(curs.fetchall())
        return frame
    return generate_pkl_cached(pkl_cache_fname, load_data)


def create_category_table():
    sql = '''
       SELECT ID_Article, Title, Path
       FROM Articles;
    '''
    con = sqlite3.connect(DBASE)
    curs = con.cursor()
    r = curs.execute(sql)
    a = np.array(curs.fetchall())
    con.close()
    paths = a[:, 2]
    splitted_paths_list = []
    for i in range(paths.shape[0]):
        splitted_path = paths[i].split("/")
        main_category, sub_category, remaining_path = "", "", ""
        try:
            main_category = splitted_path[0]
            sub_category = splitted_path[1]
            remaining_path = ''.join(splitted_path[2:])
        except IndexError:
            pass
        splitted_paths_list.append([main_category, sub_category, remaining_path])

    splitted_paths = np.array(splitted_paths_list)
    combined_array = np.concatenate([a, np.array(splitted_paths_list)], axis=1)
    category_array = combined_array[:, [0, 3, 4, 5]]
    csv = pandas.DataFrame(category_array).to_csv()
    # print(csv)

    # Add new Table with Article_Categories
    sql = '''
        CREATE TABLE IF NOT EXISTS Article_Categories (
            ID_Article integer PRIMARY KEY,
            MainCategory TEXT,
            SubCategory TEXT,
            RemainingPath TEXT
        )
    '''
    con = sqlite3.connect(DBASE)
    curs = con.cursor()
    r = curs.execute(sql)
    for i in range(len(category_array)):
        curs = con.cursor()
        article_id = category_array[i][0]
        main_category = category_array[i][1]
        sub_category = category_array[i][2]
        remaining_path = category_array[i][3]
        r = curs.execute('INSERT INTO Article_Categories(ID_Article, MainCategory, SubCategory, RemainingPath) '
                         'VALUES(?, ?, ?, ?)', (article_id, main_category, sub_category, remaining_path))
        print(curs.lastrowid)

    con.commit()
    con.close()
