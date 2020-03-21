import numpy as np
import matplotlib
import sqlite3
import pandas

DBASE = 'dataset/corpus.sqlite3'


def top_negative_articles():
    sql = '''
        SELECT Posts.ID_Article, 
        SUM(case when Posts.PositiveVotes =1 then 1 else 0 END) AS PositiveVotesCount, 
        SUM(case when Posts.NegativeVotes =1 then 1 else 0 END) AS NegativeVotesCount,
        Articles.Path,
        Articles.Title
        FROM Posts 
        LEFT JOIN Articles ON Posts.ID_Article = Articles.ID_Article
        GROUP BY Posts.ID_Article
        ORDER BY NegativeVotesCount DESC
    '''
    con = sqlite3.connect(DBASE)
    curs = con.cursor()
    r = curs.execute(sql)
    a = np.array(curs.fetchall())
    print("The 10 most disliked articles:")
    csv = pandas.DataFrame(a[:10]).to_csv(header=["ID", "PositiveVotes", "NegativeVotes", "Path", "Title"])
    print(csv)


def top_positive_articles():
    sql = '''
        SELECT Posts.ID_Article, 
        SUM(case when Posts.PositiveVotes =1 then 1 else 0 END) AS PositiveVotesCount, 
        SUM(case when Posts.NegativeVotes =1 then 1 else 0 END) AS NegativeVotesCount,
        Articles.Path,
        Articles.Title
        FROM Posts 
        LEFT JOIN Articles ON Posts.ID_Article = Articles.ID_Article
        GROUP BY Posts.ID_Article
        ORDER BY PositiveVotesCount DESC
    '''
    con = sqlite3.connect(DBASE)
    curs = con.cursor()
    r = curs.execute(sql)
    a = np.array(curs.fetchall())
    print("The 10 most liked articles:")
    csv = pandas.DataFrame(a[:10]).to_csv(header=["ID", "PositiveVotes", "NegativeVotes", "Path", "Title"])
    print(csv)


def explore_categories():
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

    con.close()
    print("done")


def main():
    top_negative_articles()
    top_positive_articles()
    explore_categories()

    print("done")


if __name__ == '__main__':
    main()
