import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import matplotlib.ticker as mticker
from datetime import datetime

# Thiết lập style chung
sns.set_theme(style="whitegrid")

# Histogram
def plot_histogram(x, title="Histogram", xlabel="", bins=30):
    plt.figure(figsize=(7,4))
    sns.histplot(x, bins=bins, kde=True, color="steelblue")
    plt.title(title, fontsize=13)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# Scatter Plot

def plot_scatter(
    x, y,
    xlabel="",
    ylabel="",
    title="Scatter Plot",
    color="viridis",
    figsize=(7, 4),
    alpha=0.45,
    add_reg=False,       # Thêm đường hồi quy
    point_size=45,
    grid=True
):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=figsize)

    # Nếu thêm regression line → dùng seaborn regression
    if add_reg:
        sns.regplot(
            x=x, y=y,
            scatter_kws={"s": point_size, "alpha": alpha},
            line_kws={"color": "red", "linewidth": 2},
            color="black"
        )
    else:
        sns.scatterplot(
            x=x, y=y,
            s=point_size,
            alpha=alpha,
            palette=color if isinstance(color, list) else None
        )

        # Nếu color chỉ là một tên colormap → dùng mapping density
        if isinstance(color, str):
            plt.scatter(
                x, y,
                c=np.linspace(0, 1, len(x)),
                cmap=color,
                s=point_size,
                alpha=alpha
            )

    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.title(title, fontsize=14)

    if grid:
        plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


def plot_heatmap(corr_matrix, labels, title="Correlation Heatmap (Triangle)", cmap="Spectral"):
    sns.set_theme(style="white")

    # Mask: True = bỏ qua (ẩn), False = hiển thị
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # ẩn tam giác trên + đường chéo

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,              # Spectral, RdYlBu, coolwarm, viridis...
        linewidths=0.5,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        square=True
    )

    plt.title(title, fontsize=15)
    plt.tight_layout()
    plt.show()

# Pie chart
def plot_pie(values, labels, title="Pie Chart"):
    plt.figure(figsize=(7, 6))

    total = sum(values)

    # Tạo label dạng:  "True (12.5%)"
    labels_pct = [f"{lbl} ({v/total*100:.1f}%)" for lbl, v in zip(labels, values)]

    # Pie không hiển thị chữ trực tiếp
    wedges, texts = plt.pie(
        values,
        labels=None,
        autopct=None,
        startangle=90
    )

    plt.title(title, fontsize=14)

    # Legend bên dưới pie
    plt.legend(
        wedges,
        labels_pct,
        title="Categories",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False
    )

    plt.tight_layout()
    plt.show()

# Line plot
def plot_line(
    y, 
    x=None,
    title="Line Plot",
    xlabel="",
    ylabel="",
    rotate_x=False,
    figsize=(12,5),
    max_xticks=10
):
    plt.figure(figsize=figsize)

    # Nếu không có x thì dùng index
    if x is None:
        x_vals = np.arange(len(y))
        plt.plot(x_vals, y, marker='o', linestyle='-', color='teal')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
        return

    # Nếu x là dạng chuỗi ngày convert sang datetime
    try:
        x_dates = [datetime.strptime(str(d), "%Y-%m-%d") for d in x]
    except:
        # fallback cho format khác
        try:
             x_dates = [parse(str(d)) for d in x] # Dùng dateutil.parser.parse linh hoạt hơn
        except Exception as e:
            print(f"Lỗi chuyển đổi ngày tháng: {e}")
            return


    # Vẽ line chart timeline
    plt.plot(x_dates, y, marker='o', linestyle='-', color='teal', alpha=0.8)

    # Bộ định dạng timeline:
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())      # tự chia tick
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # format label

    # Giảm tối đa số tick xuống max_xticks
    # SỬA: Thay mdates.MaxNLocator bằng mticker.MaxNLocator
    ax.xaxis.set_major_locator(mticker.MaxNLocator(max_xticks)) 

    if rotate_x:
        plt.xticks(rotation=45, ha='right')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# Horizontal Bar chart
def plot_bar(x_labels, heights, title="Bar Chart", xlabel="", ylabel=""):
    plt.figure(figsize=(10,6))
    sns.barplot(y=x_labels, x=heights, palette="viridis")
    
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()
    
def plot_compare_hist(
    group1, group2,
    label1="Group 1", label2="Group 2",
    title="Comparison Histogram",
    xlabel="Value",
    bins=30,
    figsize=(10,6),
    alpha1=0.6,
    alpha2=0.6,
    color1="gray",
    color2="steelblue"
):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=figsize)

    # histogram nhóm 1
    plt.hist(group1, bins=bins, alpha=alpha1, label=label1, color=color1, edgecolor="black")

    # histogram nhóm 2
    plt.hist(group2, bins=bins, alpha=alpha2, label=label2, color=color2, edgecolor="black")

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
    
def plot_boxplot(
    data_groups,
    labels,
    title="Boxplot",
    xlabel="",
    ylabel="",
    horizontal=False,
    color="#7FB3D5",      # pastel blue
    figsize=(8,5),
    showfliers=True       # hiển thị outliers
):
    plt.figure(figsize=figsize)

    # Tạo boxplot
    bp = plt.boxplot(
        data_groups,
        labels=labels,
        vert=not horizontal,   # vertical or horizontal
        patch_artist=True,
        showfliers=showfliers,
        boxprops=dict(facecolor=color, edgecolor='black'),
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        flierprops=dict(marker='o', markersize=4, alpha=0.5, markerfacecolor='gray')
    )

    # Grid nhẹ cho dễ nhìn
    plt.grid(True, linestyle="--", alpha=0.4)

    # Tên trục
    if horizontal:
        plt.xlabel(xlabel)
        plt.title(title)
    else:
        plt.ylabel(ylabel)
        plt.title(title)

    plt.tight_layout()
    plt.show()
