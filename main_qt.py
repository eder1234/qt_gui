import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from numpy import asarray
import sklearn.cluster as cluster

import matplotlib.cm as cm

def plot_clusters(df):
    clusters = df['cluster'].unique()
    n_clusters = len(clusters)
    colors = cm.rainbow(np.linspace(0, 1, n_clusters))

    plt.figure(figsize=(10, 6))
    for row in df.itertuples():
        x = row.traj[0]
        y = row.traj[1]
        plt.scatter(x, y, color=colors[row.cluster])

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

class InputWindow(QMainWindow):
    def __init__(self, master_df=None):
        super().__init__()
        self.master_df = master_df
        self.setWindowTitle("Input Window")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.input_label = QLabel("Number of clusters (int):", self)
        self.layout.addWidget(self.input_label)

        self.input_entry = QLineEdit(self)
        self.layout.addWidget(self.input_entry)

        self.action_button = QPushButton("HA clustering", self)
        self.action_button.clicked.connect(self.perform_new_action)
        self.layout.addWidget(self.action_button)
        
        self.orientation_button = QPushButton("Orientation plot", self)
        self.orientation_button.clicked.connect(self.perform_orientation)
        self.layout.addWidget(self.orientation_button)

    def perform_new_action(self):
        input_value = self.input_entry.text()
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
            plt.scatter(x-x0, y-y0)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Orientation plot')
        plt.show()

class TextFileOpenerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Text File Opener")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.file_path_1 = None
        self.file_path_2 = None
        self.master_df = None

        button_1 = QPushButton("Open trajectory file", self)
        button_1.clicked.connect(self.open_file_1)
        self.layout.addWidget(button_1)

        button_2 = QPushButton("Open motility file", self)
        button_2.clicked.connect(self.open_file_2)
        self.layout.addWidget(button_2)

        self.action_button = QPushButton("Create master_df", self)
        self.action_button.clicked.connect(self.perform_action)
        self.action_button.setEnabled(False)
        self.layout.addWidget(self.action_button)

    def open_file_1(self):
        self.file_path_1, _ = QFileDialog.getOpenFileName(filter="Text Files (*.txt)")
        print(f"File 1: {self.file_path_1}")
        self.check_files_selected()

    def open_file_2(self):
        self.file_path_2, _ = QFileDialog.getOpenFileName(filter="Text Files (*.txt)")
        print(f"File 2: {self.file_path_2}")
        self.check_files_selected()

    def check_files_selected(self):
        if self.file_path_1 is not None and self.file_path_2 is not None:
            self.action_button.setEnabled(True)
        else:
            self.action_button.setEnabled(False)

    def perform_action(self):
        print("Performing action on the selected files...")
        traj_df = pd.read_csv(self.file_path_1, names=["name", "date", "quantity", "exposure", "tracked_id", "x", "y"], engine='python')
        traj_df = trajFrame(traj_df)
        traj_df = traj_df.sort_values(by=['name', 'date', 'quantity', 'exposure', 'tracked_id']) 
        mr_df = pd.read_csv(self.file_path_2, engine='python') 
        id_ = [x for x in range(mr_df.shape[0])]
        mr_df["id"] = id_
        mr_df = mr_df.sort_values(by=['ID1', 'ID2', 'ID3', 'ID4', 'id'])

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
        input_window = InputWindow(master_df=self.master_df)

        #input_window.mainloop()

'''
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = TextFileOpenerApp()
    main_app.show()
    sys.exit(app.exec_())
'''
if __name__ == "__main__":
    app = TextFileOpenerApp()
    app.mainloop()