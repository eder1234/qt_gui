import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from numpy import asarray
import sklearn.cluster as cluster

import matplotlib.cm as cm

def plot_clusters(df):
    # Assuming that the 'cluster' column has already been added to the DataFrame
    clusters = df['cluster'].unique()
    n_clusters = len(clusters)
    colors = cm.rainbow(np.linspace(0, 1, n_clusters))

    plt.figure(figsize=(10, 6))
    for row in df.itertuples():
        x = row.traj[0]
        y = row.traj[1]
        plt.plot(x, y, color=colors[row.cluster])

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Clusters")
    plt.show()

def trajFrame(df):
    traj_df = pd.DataFrame({"name": [], "date": [], "quantity": [], "exposure": [], "tracked_id": [], "traj": []})

    df1 = df.groupby('name')
    for k1, v1 in df1:
        df2 = df1.get_group(k1)
        df3 = df2.groupby('date')
        for k3, v3 in df3:
            df4 = df3.get_group(k3)
            df5 = df4.groupby('quantity')
            for k5, v5 in df5:
                df6 = df5.get_group(k5)
                df7 = df6.groupby('exposure')
                for k7, v7 in df7:
                    df8 = df7.get_group(k7)
                    df9 = df8.groupby('tracked_id')
                    for k9, v9 in df9:
                        df10 = df9.get_group(k9)
                        x = df10['x'].to_numpy()
                        y = df10['y'].to_numpy()
                        temp_df = pd.DataFrame(
                            {"name": [k1], "date": [k3], "quantity": [k5], "exposure": [k7], "tracked_id": [k9],
                             "traj": [[x, y]]})
                        traj_df = pd.concat([temp_df, traj_df], ignore_index=True)
    return traj_df


def center_of_traj(x, y):
    max_x = np.amax(x)
    min_x = np.amin(x)
    max_y = np.amax(y)
    min_y = np.amin(y)
    x_c = np.mean([max_x, min_x])
    y_c = np.mean([max_y, min_y])
    return x_c, y_c 

def img_to_folder(m_df):
    traj_list = m_df['traj'].values.tolist()
    window_size = 60
    list_arrays = []
    for index, traj in tqdm(enumerate(traj_list)):
        list_x, list_y = zip(traj)
        x = np.asarray(list_x).squeeze()
        y = np.asarray(list_y).squeeze()
        x_c, y_c = center_of_traj(x, y)
        plt.figure(figsize=[2,2])
        plt.plot(x, y, 'k')
        plt.xlim([x_c - window_size/2, x_c + window_size/2])
        plt.ylim([y_c - window_size/2, y_c + window_size/2])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        new_id = '{:05d}'.format(index)
        plt.savefig(f'img_traj/{new_id}.png')
        plt.close()
        img = Image.open(f'img_traj/{new_id}.png').convert('L')
        img = img.resize((64, 64))
        numpydata = asarray(img).flatten()/255
        list_arrays.append(numpydata)
    m_df['img'] = list_arrays
    return m_df

class InputWindow(tk.Toplevel):
    def __init__(self, master=None, master_df=None):
        super().__init__(master)
        self.master_df = master_df
        self.title("Input Window")
        self.geometry("300x200+550+250")
        self.config(bg="#3c4245")

        custom_font = ("Arial", 15, "bold")

        self.input_label = tk.Label(self, text="Number of clusters (int):", bg="#3c4245", fg="#AEC6CF", font=custom_font)
        self.input_label.pack(pady=10)

        self.input_entry = tk.Entry(self, font=("Arial", 12))
        self.input_entry.pack(pady=10)

        self.action_button = ctk.CTkButton(self, text="HA clustering", command=self.perform_new_action, text_color='white', fg_color="#AEC6CF", bg_color="#3c4245", font=custom_font)
        self.action_button.pack(pady=10)
        
        self.orientation_button = ctk.CTkButton(self, text="Orientation plot", command=self.perform_orientation, text_color='white', fg_color="#AEC6CF", bg_color="#3c4245", font=custom_font)
        self.orientation_button.pack(pady=10)

    def perform_new_action(self):
        input_value = self.input_entry.get()
        try:
            integer_value = int(input_value)
            print(f"Input integer: {integer_value}")
            kClusters = integer_value
            list_arrays = self.master_df['img'].values.reshape(-1).tolist()
            y_labels = cluster.AgglomerativeClustering(kClusters).fit_predict(list_arrays)
            self.master_df['cluster'] = y_labels
            plot_clusters(self.master_df)
        except ValueError:
            print("Invalid input. Please enter an integer.")
    
    def perform_orientation(self):
        traj_list = self.master_df['traj'].values.tolist()

        plt.figure(figsize=(10, 10))
        for traj in traj_list:
            x = traj[0]
            x0 = x[0]
            y = traj[1]
            y0 = y[0]
            plt.plot(x-x0, y-y0)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Orientation plot')
        plt.show()

class TextFileOpenerApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Text File Opener")
        self.geometry("300x250+500+200")

        self.file_path_1 = None
        self.file_path_2 = None
        self.master_df = None

        self.config(bg="#3c4245")
        custom_font = ("Arial", 15, "bold")

        button_1 = ctk.CTkButton(self, text="Open trajectory file", command=self.open_file_1, text_color='white', fg_color="#AEC6CF", bg_color="#3c4245", font=custom_font)
        button_1.pack(pady=10)

        button_2 = ctk.CTkButton(self, text="Open motility file", command=self.open_file_2, text_color='white', fg_color="#AEC6CF", bg_color="#3c4245", font=custom_font)
        button_2.pack(pady=10)

        self.action_button = ctk.CTkButton(self, text="Create master_df", command=self.perform_action, text_color='white', fg_color="#AEC6CF", bg_color="#3c4245", font=custom_font, state='disable')
        self.action_button.pack(pady=10)

    def open_file_1(self):
        self.file_path_1 = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        print(f"File 1: {self.file_path_1}")
        self.check_files_selected()

    def open_file_2(self):
        self.file_path_2 = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        print(f"File 2: {self.file_path_2}")
        self.check_files_selected()

    def check_files_selected(self):
        if self.file_path_1 is not None and self.file_path_2 is not None:
            self.action_button.configure(state='normal')
        else:
            self.action_button.configure(state='disable')

    def perform_action(self):
        print("Performing action on the selected files...")
        traj_df = pd.read_csv(self.file_path_1, names=["name", "date", "quantity", "exposure", "tracked_id", "x", "y"], engine='python')
        traj_df = trajFrame(traj_df)
        traj_df = traj_df.sort_values(by=['name', 'date', 'quantity', 'exposure', 'tracked_id']) 
        #print(traj_df)
        mr_df = pd.read_csv(self.file_path_2, engine='python') 
        id_ = [x for x in range(mr_df.shape[0])]
        mr_df["id"] = id_
        mr_df = mr_df.sort_values(by=['ID1', 'ID2', 'ID3', 'ID4', 'id'])
        #print(mr_df)

        traj_name_list = traj_df["name"].values.tolist()
        traj_date_list = traj_df["date"].values.tolist()
        traj_quantity_list = traj_df["quantity"].values.tolist()
        traj_exposure_list = traj_df["exposure"].values.tolist()
        traj_tracked_id_list = traj_df["tracked_id"].values.tolist()
        traj_traj_list = traj_df["traj"].values.tolist()
        mr_VCL_list = mr_df["VCL"].values.tolist()
        mr_VAP_list = mr_df["VAP"].values.tolist()
        mr_VSL_list = mr_df["VSL"].values.tolist()
        mr_LIN_list = mr_df["LIN"].values.tolist()
        mr_STR_list = mr_df["STR"].values.tolist()
        mr_WOB_list = mr_df["WOB"].values.tolist()
        mr_BeatCross_list = mr_df["BeatCross"].values.tolist()
        mr_ALH_list = mr_df["ALH"].values.tolist()

        master_dict = {"name":traj_name_list, "date":traj_date_list, "quantity":traj_quantity_list, 
               "exposure":traj_exposure_list, "tracked_id":traj_tracked_id_list, "traj":traj_traj_list,
              "VCL":mr_VCL_list, "VAP":mr_VAP_list, "VSL":mr_VSL_list, "LIN":mr_LIN_list, "STR":mr_STR_list,
              "WOB":mr_WOB_list, "BeatCross":mr_BeatCross_list, "ALH":mr_ALH_list}
        
        master_df = pd.DataFrame(master_dict)
        master_df = img_to_folder(master_df)
        self.master_df = master_df
        input_window = InputWindow(self, self.master_df)

        input_window.mainloop()

if __name__ == "__main__":
    app = TextFileOpenerApp()
    app.mainloop()
