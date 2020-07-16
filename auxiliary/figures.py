import matplotlib.pyplot as plt

def plot_educ_against_yob(df):

    mean_educ = df.groupby(['YOB', 'QOB'])['EDUC'].mean()

    df_index = mean_educ.index.to_frame().reset_index(drop = True)
    x_values = df_index.apply(lambda x: x[0] + x[1] * 0.25 - 0.25, axis = 1)

    y_values = mean_educ.values

    _, ax = plt.subplots(1, 1, figsize = (10, 5))
    ax.plot(x_values, y_values, color = 'k', marker = 's')

    point_annotations = ['1', '2', '3', '4']

    for i in range(len(x_values)):
        ax.annotate(point_annotations[i % 4], (x_values[i] - 0.075, y_values[i] - 0.075))

    ax.set_xlabel('Year of Birth')
    ax.set_ylabel('Years Of Completed Education')

    plt.plot(x_values, y_values)

def plot_bar_detrended_educ(df):

    df_index = df.index.to_frame().reset_index(drop = True)

    x_values = df_index.apply(lambda x: x[0] + x[1] * 0.25 - 0.25, axis = 1).to_numpy()
    y_values = df['DTRND'][:len(x_values)].to_numpy()

    _, ax = plt.subplots(1,1)

    ax.bar(x_values, y_values, width = 0.25, color= ['#000000', '#404040', '#7f7f7f', '#bfbfbf'])

    point_annotations = ['1', '2', '3', '4']
    for i in range(len(x_values)):
        ax.annotate(point_annotations[i % 4], (x_values[i] - 0.075, y_values[i]))

    ax.set_xlabel('Year of Birth')
    ax.set_ylabel('Schooling Differential')

def plot_log_wkly_earnings_by_qob(df):

    mean_lwklywge = df.groupby(['YOB', 'QOB'])['LWKLYWGE'].mean()

    df_index = mean_lwklywge.index.to_frame().reset_index(drop = True)
    x_values = df_index.apply(lambda x: x[0] + x[1] * 0.25 - 0.25, axis = 1)

    y_values = mean_lwklywge.values

    _, ax = plt.subplots(1, 1)
    ax.plot(x_values, y_values, color = 'k', marker = 's')

    point_annotations = ['1', '2', '3', '4']

    for i in range(len(x_values)):
        ax.annotate(point_annotations[i % 4], (x_values[i], y_values[i]))

    ax.set_xlabel('Year of Birth')
    ax.set_ylabel('Log Weekly Earnings')

    plt.plot(x_values, y_values)