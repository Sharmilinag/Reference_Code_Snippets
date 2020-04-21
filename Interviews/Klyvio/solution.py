import pandas as pd
import numpy as np
import sys
from datetime import datetime
import matplotlib.pyplot as plt
# import researchpy as rp
from scipy import stats
from sklearn.metrics import confusion_matrix

datetime_parser = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv('screening_exercise_orders_v201810_NEW.csv', parse_dates=['date'], date_parser=datetime_parser)


# df['date'] = df['date'].apply(lambda x: pd.datetime.strptime(x, '%Y-%m-%d %I:%M:%S %p'))
print df.head()
# print df.columns.values
# print df.shape
# print df.dtypes
# print df.date.min(), df.date.max()
# sys.exit()

# print len(df.customer_id.unique())

#   ## Question A
# def convert_datetime_format(x):
#     return datetime.strftime(x, "%Y-%m-%d %I:%M:%S %p")
#
#
# grouped = df.groupby(by=['customer_id', 'gender'], as_index=False)
# qs_a = grouped.agg({"date": "max", "value": "size"})
# qs_a.rename({"date": "most_recent_order_date", "value": "order_count"}, axis=1, inplace=True)
# qs_a['most_recent_order_date'] = qs_a['most_recent_order_date'].apply(convert_datetime_format)
# print qs_a[qs_a['customer_id'] == 1034]
# print type(qs_a)
# print qs_a.dtypes

#   ##  Question B
# df["week_number_of_year"] = df["date"].dt.week
# grouped = df.groupby(by=['week_number_of_year'], as_index=False)
# qs_b = grouped.agg({"date": "size"})
# qs_b.rename({'date': 'count_of_orders'}, axis=1, inplace=True)
# ax = qs_b.plot.bar(x='week_number_of_year', y='count_of_orders', legend=False)  #, figsize=(20, 20))
# for p in ax.patches:
#     ax.annotate(str(p.get_height()), (p.get_x() - 0.08, p.get_height() + 2.5), fontsize=6, fontweight='bold')
# ax.set_xlabel("Week Number of Year", fontweight='bold', fontsize=14)
# ax.set_ylabel("Count of Orders", fontweight='bold', fontsize=14)
# ax.set_title('Count of Orders per Week', fontweight='bold', fontsize=18)
# plt.show()

  ##  Question C
order_gen_0 = df[df['gender'] == 0]['value']
order_gen_1 = df[df['gender'] == 1]['value']
print order_gen_0.mean(), order_gen_1.mean()
print order_gen_0.shape, order_gen_1.shape
#   ##  Assumptions of Independent t-test
# The samples are independently and randomly drawn
# The distribution of the residuals between the two groups should follow the normal distribution
# The variances between the two groups are equal
#   ##   Check Assumptions for the independent t-test,
# the assumptions of the t-test need to be checked to see if the t-test results can be trusted
#   ##  1.  Homogeneity of variances
print stats.levene(order_gen_0, order_gen_1)
#   The p-value is > 0.05, This means that the test is significant. This means there is no homogeneity of variances
#   ##  Hence, moving on to Welch\'s t-test as there is violation in the assumption of equality of variances
#   ##  Assumptions of Welch\'s t-test
# The independent variable (IV) is categorical with at least two levels (groups)
# The dependent variable (DV) is continuous which is measured on an interval or ratio scale
# The distribution of the two groups should follow the normal distribution
#   ##  The first and the second  assumption is evident from the data
#   ##   Checking for Normal Distribution
# 1. The Shapiro-Wilk test evaluates how likely a data sample was drawn from a Gaussian distribution
#   ##  Testing for Gender 0
stat_shapiro_0, p_shapiro_0 = stats.shapiro(order_gen_0)
print stat_shapiro_0, p_shapiro_0
#   ##  Testing for Gender 1
stat_shapiro_1, p_shapiro_1 = stats.shapiro(order_gen_1)
print stat_shapiro_1, p_shapiro_1
# 2. Checking for Normality using the p-p plot
#   ##  Testing for Gender 0
stats.probplot(order_gen_0, plot=plt)
plt.title('Orders of Gender 0 P-P Plot')
plt.show()
#   ##  Testing for Gender 1
stats.probplot(order_gen_1, plot=plt)
plt.title('Orders of Gender 1 P-P Plot')
plt.show()
# 3. Checking using histograms
#   ##  Testing for Gender 0
order_gen_0.plot(kind="hist", title= "Order Values of customers with Gender 0")
plt.xlabel("order value")
plt.show()
#   ##  Testing for Gender 1
order_gen_1.plot(kind="hist", title= "Order Values of customers with Gender 1")
plt.xlabel("order value")
plt.show()
#   ##  Thus we see that the samples are not drawn from a Normal Distribution.
#   ##  Hence, we need to do Non-Parametric Test. We go for the Mann-Whitney U test
#   ##  The Null Hypothesis : the Sample distributions are equal
stat, p = stats.mannwhitneyu(order_gen_0, order_gen_1)
print('Statistics=%.3f, p=%.3f' % (stat, p))
#   ##  Since, the p-value > 0.05, hence we accept the Null Hypothesis, thus the Sample distributions are equal
# TODO : significance of mean difference

#   ##  Question D
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print "    " + empty_cell,
    for label in labels:
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "    %{0}s".format(columnwidth) % label1,
        for j in range(len(labels)):
            cell = "%{0}.0f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print cell,
        print


cm = confusion_matrix(df['gender'], df['predicted_gender'], labels=[0, 1])
print cm
print_cm(cm, ['0', '1'])
