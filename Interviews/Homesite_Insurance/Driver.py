import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import seaborn as sb
from relativeImp import relativeImp
import chart_studio.plotly
from dominance_analysis import Dominance
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn import preprocessing
from scipy import stats
import statsmodels.api as sm
cat_cols = ['PolicyStatus', 'CancellationReason', 'SoldByChannel',
            'Property: Top 10 States', 'Region' 'PolicyPaymentType',
            'Number of Days Cancelled Policy was Active']

explored_cols = []

FILTER = 'Benchmark'


def reformat_date(df_raw):
    global explored_cols
    print(df_raw.shape)
    print(df_raw.columns.values)
    print('*******************************')
    df_raw = df_raw[df_raw['PartnerName'] == FILTER]
    # df_raw = df_raw[df_raw['PolicyStatus'] == 'Cancelled']
    df_raw.drop(columns=['PartnerName', 'Number of Days Cancelled Policy was Active'], inplace=True)
    print(df_raw.shape)
    print(df_raw.columns.values)
    print('*******************************')
    df = pd.get_dummies(df_raw)
    df.drop(['Sum of Policies','PolicyStatus_Cancelled','PolicyStatus_Active', 'CancellationReason_Company-Initiated',
             'CancellationReason_Customer-Initiated', 'CancellationReason_Null'], axis=1, inplace=True)
    # df.drop('Sum of Policies', axis=1, inplace=True)
    print(df.shape)
    print(df.columns.values)
    explored_cols = list(df.columns.values)
    print('*******************************\n\n')
    return df


def lasso_feat_imp(df):
    X = df.drop('PolicyStatus_Cancelled', axis=1)
    y = df['PolicyStatus_Cancelled']
    reg = LassoCV()
    reg.fit(X, y)
    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" % reg.score(X, y))
    coef = pd.Series(reg.coef_, index=X.columns)
    print("Lasso picked " + str(sum(coef != 0))
          + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")
    imp_coef = coef.sort_values()
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind="barh")
    plt.title("Feature importance using Lasso Model")
    plt.show()


def feat_imp(df):
    forest = DecisionTreeRegressor(max_depth=5)
        # RandomForestRegressor(max_depth=2, random_state=0,
        #                          n_estimators=100)
    X = df.drop(['PolicyStatus_Cancelled','Cancellation Rate'], axis=1)
    y = df['Cancellation Rate']
    forest.fit(X, y)
    print(dict(zip(X.columns, forest.feature_importances_)))
    return

    feat_list = list(X.columns.values)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, feat_list[int(indices[f])], importances[indices[f]]))
        # print(feat_list[int(indices[f])])

    # Plot the feature importances of the forest
    # plt.figure()
    # plt.title("Feature importances")
    # plt.barh(range(X.shape[1]), importances[indices],
    #         color="r", align="center")
    #
    # plt.yticks(range(X.shape[1]), indices)
    # plt.ylim([-1, X.shape[1]])
    # plt.show()
    # plt.savefig(FILTER+'_feat_imp.png')

    ############ Sharmili ##############
    df_test = pd.DataFrame({'var_list': feat_list, 'importances': importances})
    df_test.sort_values('importances', inplace=True)
    df_test.plot(kind='barh', y='importances', x='var_list', color='r', fontsize=8.3,
                 legend=False)
    plt.show()
    plt.savefig(FILTER + '_feat_imp.png')
    ############ Sharmili ##############


def check_linearity(df):
    plt.scatter(df['SoldByChannel_Agents'], df['PolicyStatus_Cancelled'], color='red')
    # plt.title('Stock Index Price Vs Interest Rate', fontsize=14)
    plt.xlabel('Interest Rate', fontsize=14)
    plt.ylabel('Stock Index Price', fontsize=14)
    plt.grid(True)
    plt.show()


def calc_relative_imp(df):
    # yName = 'PolicyStatus_Cancelled'
    # xNames = explored_cols
    # xNames.remove('PolicyStatus_Cancelled')
    # xNames.remove('PolicyStatus_Active')
    # df.replace([np.inf, -np.inf], np.nan)
    # df.fillna(-1, inplace=True)
    # df_raw = pd.read_excel('AnalystInterview Case.xlsx', sheet_name='Data', usecols=['PolicyStatus', 'SoldByChannel',
    #         'Property: Top 10 States', 'Region' ,'PolicyPaymentType', 'PartnerName',
    #         'Number of Days Cancelled Policy was Active'])
    # df_raw = df_raw[df_raw['PartnerName'] == 'Company X']
    dominance_classification = Dominance(data=df, target='PolicyStatus_Cancelled', objective=0, pseudo_r2="mcfadden",
                                         data_format=0)
    print(dominance_classification)


def calculate_corr(df):
    pearsoncorr = df.corr(method='pearson')
    print(pearsoncorr)
    plot_heat_map_graph(corr_matrix=pearsoncorr)


def plot_heat_map_graph(corr_matrix):
    graph = sb.heatmap(corr_matrix,
                       xticklabels=corr_matrix.columns,
                       yticklabels=corr_matrix.columns,
                       # cmap='RdBu_r',
                       annot=False,
                       linewidth=0.5)
    # graph.set_yticklabels(labels=corr_matrix.columns)#, rotation=45)
    # graph.set_xticklabels(labels=corr_matrix.columns)#, rotation=45)
    plt.show()


def plot_pairplot(df):
    sb.set(style="ticks", color_codes=True)
    sb.pairplot(df)
    plt.savefig(FILTER + '_pairplot.png')


def check_dist(df):
    df = df[df['Cancellation Rate'] != 0]
    print(df['Cancellation Rate'].unique())
    ####### output distribution #######
    # plt.hist(df['Cancellation Rate'], color='blue', edgecolor='black',
    #          bins=int(180 / 5))
    # plt.show()
    ###### convert to normalized ######

    # Cancellation_Rate = df[['Cancellation Rate']].values.astype(float)
    df['Cancellation Rate'] = (df['Cancellation Rate'] - df['Cancellation Rate'].mean()) / df['Cancellation Rate'].std()
    print(df['Cancellation Rate'].unique())
    # Cancellation_Rate = df[['Cancellation Rate']].values.astype(float)
    # min_max_scaler = preprocessing.MinMaxScaler()
    # Cancellation_Rate_scaled = min_max_scaler.fit_transform(Cancellation_Rate)
    # df_normalized = pd.DataFrame(Cancellation_Rate_scaled)
    # print(Cancellation_Rate_scaled.unique())
    print("normalized data")
    # print(df_normalized)
    plt.hist(df['Cancellation Rate'], range= [0.1, 0.2], color='blue', edgecolor='black',
             bins=int(180 / 5))
    plt.show()
def point_bi_correlation(df):
    Cancellation_Rate = df['Cancellation Rate'].to_list()
    # SoldByChannel_Agents = df['SoldByChannel_Agents'].to_list()
    SoldByChannel_Call_Center = df['SoldByChannel_Call Center'].to_list()
    # SoldByChannel_Referral = df['SoldByChannel_Referral'].to_list()
    # SoldByChannel_Website = df['SoldByChannel_Website'].to_list()
    States_AZ = df['Property: Top 10 States_AZ'].to_list()
    States_CA = df['Property: Top 10 States_CA'].to_list()
    States_LA = df['Property: Top 10 States_LA'].to_list()
    States_MA = df['Property: Top 10 States_MA'].to_list()
    States_NJ = df['Property: Top 10 States_NJ'].to_list()
    States_NY = df['Property: Top 10 States_NY'].to_list()
    States_Others = df['Property: Top 10 States_Others'].to_list()
    States_TN = df['Property: Top 10 States_TN'].to_list()
    States_TX = df['Property: Top 10 States_TX'].to_list()
    States_WA = df['Property: Top 10 States_WA'].to_list()
    Region_Central = df['Region_Central'].to_list()
    Region_Northeast = df['Region_Northeast'].to_list()
    Region_West = df['Region_West'].to_list()
    PolicyPaymentType_AutoPay = df['PolicyPaymentType_AutoPay'].to_list()
    PolicyPaymentType_Manual = df['PolicyPaymentType_Manual'].to_list()
    # print(stats.pointbiserialr(Cancellation_Rate, SoldByChannel_Agents))
    print(stats.pointbiserialr(Cancellation_Rate, SoldByChannel_Call_Center))
    # print(stats.pointbiserialr(Cancellation_Rate, SoldByChannel_Referral))
    # print(stats.pointbiserialr(Cancellation_Rate, SoldByChannel_Website))
    print(stats.pointbiserialr(Cancellation_Rate, States_AZ))
    print(stats.pointbiserialr(Cancellation_Rate, States_CA))
    print(stats.pointbiserialr(Cancellation_Rate, States_LA))
    print(stats.pointbiserialr(Cancellation_Rate, States_MA))
    print(stats.pointbiserialr(Cancellation_Rate, States_NJ))
    print(stats.pointbiserialr(Cancellation_Rate, States_NY))
    print(stats.pointbiserialr(Cancellation_Rate, States_Others))
    print(stats.pointbiserialr(Cancellation_Rate, States_TN))
    print(stats.pointbiserialr(Cancellation_Rate, States_TX))
    print(stats.pointbiserialr(Cancellation_Rate, States_WA))
    print(stats.pointbiserialr(Cancellation_Rate, Region_Central))
    print(stats.pointbiserialr(Cancellation_Rate, Region_Northeast))
    print(stats.pointbiserialr(Cancellation_Rate, Region_West))
    print(stats.pointbiserialr(Cancellation_Rate, PolicyPaymentType_AutoPay))
    print(stats.pointbiserialr(Cancellation_Rate, PolicyPaymentType_Manual))


def regression_analysis(df):
    X = df[[ 'SoldByChannel_Call Center',
  'Property: Top 10 States_AZ', 'Property: Top 10 States_CA',
 'Property: Top 10 States_LA', 'Property: Top 10 States_MA',
 'Property: Top 10 States_NJ', 'Property: Top 10 States_NY',
 'Property: Top 10 States_Others', 'Property: Top 10 States_TN',
 'Property: Top 10 States_TX', 'Property: Top 10 States_WA',
 'Region_Central', 'Region_Northeast', 'Region_West',
 'PolicyPaymentType_AutoPay', 'PolicyPaymentType_Manual']]
    y = df['Cancellation Rate']
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    print(model.summary())

def main():
    df_raw = pd.read_excel('AnalystInterview Case.xlsx', sheet_name='Data')
    df = reformat_date(df_raw)
    # point_bi_correlation(df)
    regression_analysis(df)
    # check_dist(df)
    # plot_pairplot(df)
    # print(df_raw.describe())
    # calculate_corr(df)
    # calc_relative_imp(df=df)
    # feat_imp(df)
    # check_linearity(df)
    # lasso_feat_imp(df)


if __name__ == "__main__":
    main()
