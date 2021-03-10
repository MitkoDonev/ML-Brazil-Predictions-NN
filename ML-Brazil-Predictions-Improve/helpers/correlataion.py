import seaborn as sns
import matplotlib.pyplot as plt


def find_correlated_columns(df, target):
    # find correlation between columns
    corr = df.corr()
    # make values absolute and sort
    corr = abs(corr.sort_values(by=[target], ascending=False))

    # plot corr
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr.round(2), annot=True, cmap=plt.cm.Reds)
    plt.show()

    # select highly correlated features
    corr_target = corr[target]

    relevant_features = corr_target[corr_target > 0.1]

    print(relevant_features)

    list_columns = []

    for col, val in relevant_features.iteritems():
        if val < 1:
            list_columns.append(col)

    return list_columns
