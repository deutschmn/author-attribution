import sqlite3
import matplotlib.pyplot as plt
import pandas
import sqlalchemy
import dbase_helper
import plt_helper
import numpy as np
import datetime


def generate_joined_category_articles_frame():
    article_frame = dbase_helper.query_to_data_frame(
        '''
        SELECT Articles.ID_Article, Articles.Title, Articles.publishingDate, Article_Categories.MainCategory, 
        Article_Categories.SubCategory, Article_Categories.RemainingPath, Articles.Body
        FROM Articles LEFT JOIN Article_Categories ON Article_Categories.ID_Article = Articles.ID_Article;
        ''',
        "joined_category_articles.pkl"
    )
    article_frame.columns = ['Article_ID', 'Title', 'PublishingDate', 'MainCategory', 'SubCategory', 'RemainingPath',
                             'Body']
    article_frame['PublishingDate'] = pandas.to_datetime(article_frame['PublishingDate'])
    return article_frame


def category_analysis():
    frame = dbase_helper.get_pandas_from_table("Article_Categories")
    main_categories = np.array(frame.MainCategory)
    plt_helper.plot_histogram_distinct("Main Category Distribution", main_categories)

    for current_main_category in np.unique(main_categories):
        main_category_data = frame[frame['MainCategory'] == current_main_category]
        main_category_sub_categories = main_category_data["SubCategory"]
        plt_helper.plot_histogram_distinct("Category Distribution " + current_main_category,
                                           main_category_sub_categories)

    article_frame = generate_joined_category_articles_frame()

    # Do stuff with articles by year
    years = np.array(article_frame.sort_values(by='PublishingDate')['PublishingDate'].dt.year)
    plt_helper.plot_histogram_distinct("Article count over years", years)
    min_year = years[0]
    max_year = years[-1]

    # We actually only have articles for 2015 and 2016 all others years have only one article
    min_year = 2015
    for year in range(min_year, max_year + 1):
        year_articles = article_frame[article_frame['PublishingDate'].dt.year == year]
        main_categories = np.array(year_articles.MainCategory)
        plt_helper.plot_histogram_distinct("Main Category Distribution " + str(year), main_categories)
        for current_main_category in np.unique(main_categories):
            main_category_data = year_articles[year_articles['MainCategory'] == current_main_category]
            main_category_sub_categories = main_category_data["SubCategory"]
            plt_helper.plot_histogram_distinct("Category Distribution " + current_main_category + str(year),
                                               main_category_sub_categories)

    # Time & Day analysis
    newsroom_articles = article_frame[article_frame.MainCategory == 'Newsroom']
    pandas.Series(newsroom_articles.PublishingDate.dt.hour).plot.hist(alpha=0.8, bins=list(range(0, 24)), rwidth=0.8)
    plt_helper.save_and_show_plot("Time Distribution Newsroom")
    plt_helper.plot_day_histogram("Day Distribution Newsroom", newsroom_articles.PublishingDate.dt.weekday)

    sub_categories = np.unique(newsroom_articles.SubCategory)
    for category in sub_categories:
        pandas.Series(
            newsroom_articles[newsroom_articles.SubCategory == category].PublishingDate.dt.hour).plot.hist(
            alpha=0.8, bins=list(range(0, 24)), rwidth=0.8)
        plt_helper.save_and_show_plot("Time Distribution " + str(category))

        days = newsroom_articles[newsroom_articles.SubCategory == category].PublishingDate.dt.weekday
        plt_helper.plot_day_histogram("Day Distribution " + str(category), days)
    print("done")


def generate_joined_rating_articles_frame():
    sql = '''
            SELECT Posts.ID_Article, 
            Articles.Title,
            SUM(case when Posts.PositiveVotes =1 then 1 else 0 END) AS PositiveVotesCount, 
            SUM(case when Posts.NegativeVotes =1 then 1 else 0 END) AS NegativeVotesCount,
            Article_Categories.MainCategory,
            Article_Categories.SubCategory,
            Article_Categories.RemainingPath,
            Articles.Body
            FROM Posts 
            LEFT JOIN Articles ON Posts.ID_Article = Articles.ID_Article
            LEFT JOIN Article_Categories ON Posts.ID_Article = Article_Categories.ID_Article
            GROUP BY Posts.ID_Article;
        '''
    frame = dbase_helper.query_to_data_frame(sql, "joined_rating_articles.pkl")
    frame.columns = ["ID_Article", "Title", "PositiveVotesCount", "NegativeVotesCount", "MainCategory", "SubCategory",
                     "RemainingPath", "Body"]
    return frame


def rating_analysis():
    frame = generate_joined_rating_articles_frame()
    main_category_votes = frame[["PositiveVotesCount", "NegativeVotesCount", "MainCategory"]].groupby(
        by="MainCategory").sum()
    main_category_votes.plot(kind='bar')
    plt_helper.save_and_show_plot("Votes for Posts per Main Category")

    newsroom_data = frame[frame.MainCategory == "Newsroom"]
    newsroom_votes = newsroom_data[["PositiveVotesCount", "NegativeVotesCount", "SubCategory"]].groupby(
        by="SubCategory").sum()
    newsroom_votes.plot(kind='bar')
    plt_helper.save_and_show_plot("Votes for Posts per Newsroom Category")
    print("done")


if __name__ == '__main__':
    rating_analysis()
    category_analysis()
