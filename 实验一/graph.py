from matplotlib import pyplot as plt

# 设置字体，解决中文无法识别的问题
# 图表标题
font_title = {
    'family': 'Microsoft Yahei',
    'weight': 'regular',
    'size': 12
}

# 坐标轴标题
font_label = {
    'family': 'Microsoft Yahei',
    'weight': 'regular',
    'size': 10
}

# 图例
font_legend = {
    'family': 'Microsoft Yahei',
    'weight': 'regular',
    'size': 6
}


def init_graph(xlabel, ylabel, title, dpi=150, style="seaborn-bright"):
    # 设置清晰度
    plt.figure(dpi=dpi)

    # 设置样式
    plt.style.use(style)

    # 添加x，y轴名称
    plt.xlabel(fontdict=font_label, xlabel=xlabel)
    plt.ylabel(fontdict=font_label, ylabel=ylabel)

    # 添加标题
    plt.title(title, font=font_title)


def draw_graph(legend=False, save=False, filename="picture.jpg", show=False):
    # 是否添加图例
    if legend:
        plt.legend(prop=font_legend, loc='upper right', frameon=False)

    # 是否保存
    if save:
        plt.savefig("./figures/" + filename)

    if show:
        plt.show()
