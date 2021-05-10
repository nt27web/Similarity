import json
import requests

from scipy import sparse
import sklearn.metrics.pairwise as pw
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from IPython.display import display

from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def _recommend():

    # data = pd.read_csv('walmart_20191_data.csv')
    # f_data = pd.DataFrame(data, columns=['Product Name', 'List Price', 'Brand', 'Category', 'Postal Code', 'Available'])
    # f_data['Total Reviews'] = np.random.randint(0, 100, size=len(f_data))
    # f_data['Rating'] = np.random.randint(1, 5, size=len(f_data))
    # pd = f_data.to_csv('walmart_2019_data.csv', index=True)
    # return 0

    data = pd.read_csv('walmart_2019_data.csv')
    f_data = pd.DataFrame(data, columns=['Product Name', 'List Price', 'Category', 'Total Reviews', 'Rating'])

    m = f_data['Rating'].quantile(0.6)
    m_data = f_data[f_data['Rating'] >= m]

    pivot_item_based = pd.pivot_table(m_data,
                                      index='Product Name',
                                      columns=['List Price'], values=['Rating', 'Category'])

    print(pivot_item_based.head(10))
    return 0
    sparse_pivot = sparse.csr_matrix(pivot_item_based.fillna(0))

    # print(sparse_pivot)
    # print(sparse_pivot.shape)
    # return 0
    recommender = pw.cosine_similarity(sparse_pivot)
    recommender_df = pd.DataFrame(recommender,
                                  columns=pivot_item_based.index,
                                  index=pivot_item_based.index)

    # print(recommender_df.head())
    # return 0
    ## Product Rating Based Cosine Similarity
    # cosine_df = pd.DataFrame(
    #     recommender_df['La Costena Chipotle Peppers, 7 OZ (Pack of 12)'].sort_values(ascending=False))

    # cosine_df = pd.DataFrame(
    #     recommender_df['Otto Cap Brushed Cotton Twill High Crown Golf Style Caps - Hat / Cap for Summer, Sports, Picnic, Casual wear and Reunion etc'].sort_values(ascending=False))

    # cosine_df = pd.DataFrame(
    #     recommender_df[
    #         'Pristine Blue Pristine Power Non-Chlorine Shock for Pools and Spas'].sort_values(ascending=False))

    # cosine_df = pd.DataFrame(
    #     recommender_df[
    #         'Creative Covers For Golf Tom Driver Headcover'].sort_values(ascending=False))

    # cosine_df = pd.DataFrame(
    #     recommender_df[
    #         'Tie-Me-Not Curly No-Tie Twister Shoelaces, 2 Pairs White'].sort_values(ascending=False))



    # cosine_df = pd.DataFrame(
    #     recommender_df[
    #         'Veridian 2-Second Digital Thermometer'].sort_values(ascending=False))

    cosine_df = pd.DataFrame(
        recommender_df[
            'Paragon Popcorn Butter Bags'].sort_values(ascending=False))

    cosine_df.reset_index(level=0, inplace=True)
    cosine_df.columns = ['Product Name', 'cosine_similarity']
    print(cosine_df.head(10))

    # similar = cosine_df['cosine_sim'] > 0.5
    # cosine_df_r = cosine_df[similar]
    # print(cosine_df_r)

    # data = pd.read_csv('HR-Employee-Attrition.csv')

    # display(data.shape)
    # display(data.head(10))
    # display(data.isnull().sum())

    # display(data['Age'].describe())
    # display(data['DailyRate'].describe())
    # display(data['EducationField'].unique())
    #
    # display(data['YearsAtCompany'].describe())
    # display(data['YearsInCurrentRole'].describe())
    # display(data['YearsSinceLastPromotion'].describe())
    # display(data['YearsWithCurrManager'].describe())

    # display(len(f_data['Category'].unique()))
    # display(len(data))

    # return 0
    # display(len(data))

    # f_data = pd.DataFrame(data, columns=['Attrition', 'DailyRate', 'EducationField', 'YearsAtCompany'
    #     , 'YearsInCurrentRole',	'YearsSinceLastPromotion',	'YearsWithCurrManager'
    # ])

    # display(f_data.head())

    # m_data = f_data[f_data['Attrition'] == 'Yes']
    # f_data = m_data.drop( ['Attrition'], axis=1)

    # display(len(f_data))
    #
    # return 0

    # display(f_data.head())
    # return 0
    # display(data.shape)
    # X = f_data
    # y = f_data['Gender']
    # le = LabelEncoder()
    # X['Gender'] = le.fit_transform(X['Gender'])
    # y = le.transform(y)
    #
    # X = f_data
    # y = f_data['Department']
    # le = LabelEncoder()
    # X['Department'] = le.fit_transform(X['Department'])
    #
    # y = le.transform(y)

    # display(f_data['EducationField'].unique())
    # X = f_data
    # y = f_data['EducationField']
    # le = LabelEncoder()
    # X['EducationField'] = le.fit_transform(X['EducationField'])
    # y = le.transform(y)
    # display(y)

    # cols = X.columns
    #
    # ms = MinMaxScaler()
    # X = ms.fit_transform(X)
    #
    # X = pd.DataFrame(X, columns=[cols])

    # display(X.head())


    # cs = []
    # for i in range(1, 12):
    #     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    #     kmeans.fit(X)
    #     cs.append(kmeans.inertia_)
    # plt.plot(range(1, 12), cs)
    # plt.title('The Elbow Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('CS')
    # plt.show()

    # calculate_k_value(X)
    # return 0

    # find_k_means(X, y, 2)

    return 0


def calculate_k_value(x):
    error = []
    for i in range(1, 12):
        kmeans = KMeans(n_clusters=i).fit(x)
        kmeans.fit(x)
        error.append(kmeans.inertia_)
    import matplotlib.pyplot as plt
    plt.plot(range(1, 12), error)
    plt.title('Elbow method')
    plt.xlabel('No of clusters')
    plt.ylabel('Error')
    plt.show()


def find_k_means(x, y, k):
    k_means = KMeans(n_clusters=3, random_state=0)
    y_k_means = k_means.fit_predict(x)
    labels = k_means.labels_
    # check how many of the samples were correctly labeled
    correct_labels = sum(y == labels)
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
    print('Accuracy score: {0:0.2f} %'.format((correct_labels *100)/ float(y.size)))
    # display(x.shape)

    plt.scatter(x['YearsAtCompany'],x['YearsWithCurrManager'], c=y_k_means, cmap='rainbow')
    plt.title('Scatter Plot with K = 3')
    plt.xlabel('Number of Years at Company')
    plt.ylabel('Years With Current Manager')
    plt.show()


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+Shift+B to toggle the breakpoint.


if __name__ == '__main__':
    # print_hi('PyCharm')
    _recommend()

