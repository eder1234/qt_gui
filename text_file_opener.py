import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
import pandas as pd

def open_file_1():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    #print(f"File 1: {file_path}")
    #file_path = "traj-partial.txt"
    traj_df = pd.read_csv(file_path, names=["name", "date", "quantity", "exposure", "tracked_id", "x", "y"], engine='python')
    traj_df = trajFrame(traj_df)
    traj_df = traj_df.sort_values(by=['name', 'date', 'quantity', 'exposure', 'tracked_id']) 
    print(traj_df)

def open_file_2():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    mr_df = pd.read_csv(file_path, engine='python') 
    id_ = [x for x in range(mr_df.shape[0])]
    mr_df["id"] = id_
    mr_df = mr_df.sort_values(by=['ID1', 'ID2', 'ID3', 'ID4', 'id'])
    print(mr_df)

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

root = tk.Tk()
#change the background color
root.config(bg="#3c4245")
root.title("Text File Opener")

# Set window size and position
root.geometry("300x200+500+200")

# Button 1 to open text file
button_1 = ctk.CTkButton(root, text="Open trajectory file", command=open_file_1, font= ("Arial", 15, "bold"), text_color = 'navy', fg_color="#AEC6CF", bg_color="#3c4245")

button_1.pack(pady=20)

# Button 2 to open text file
button_2 = ctk.CTkButton(root, text="Open motility file", command=open_file_2, font= ("Arial", 15, "bold"), text_color = 'navy', fg_color="#AEC6CF", bg_color="#3c4245")

button_2.pack(pady=20)

root.mainloop()
