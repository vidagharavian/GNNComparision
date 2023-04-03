import os

import pandas as pd

import matplotlib.pyplot as plt


def plot_generation_accuracy(seed, dimensions, data_set, model_name, base_lines):
    save_path = f"./logs/{data_set}/generationSeed{seed}Dimensions{dimensions}"
    directories = [f.path for f in os.scandir(save_path) if f.is_dir()]
    base_line_x = {}
    base_line_y = {}
    for directory in directories:
        try:
            for base_line in base_lines:
                path = directory + f"/{model_name}_{base_line}_accuracy.csv"
                df = pd.read_csv(path)
                try:
                    base_line_x[base_line].append(df['generation'].values[0])
                    base_line_y[base_line].append(df['accuracy'].values[0])
                except:
                    base_line_x[base_line] = [df['generation'].values[0]]
                    base_line_y[base_line] = [df['accuracy'].values[0]]
        except:
            pass
    fig = plt.figure()
    ax = plt.axes()
    plt.title(f"Generation Accuracy Dimension {dimensions}")
    plt.xlabel("generation")
    plt.ylabel("accuracy")

    for key, value in base_line_x.items():
        df = pd.DataFrame({"x": value, "y": base_line_y[key]}).sort_values(by="x")
        plt.plot(df['x'], df['y'], label=key)
        df.to_csv(key + ".csv")

    plt.legend()
    plt.axis('tight')
    plt.savefig(f"generationSeed{seed}Dimensions{dimensions}Accuracy.png")
    plt.show()

    print(base_line_x)


plot_generation_accuracy(10, 2, 'mine', "DIGRAC", base_lines=["dist_latest", "innerproduct_latest"])
