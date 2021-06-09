import os, shutil
import pandas as pd
import numpy as np
from Model.Simulation.experiment import Experiment
from Model.Simulation.parse import Parse


class General:

    @staticmethod
    def get_ol_ofs_ruintimes(base_path=None):
        windows = ['1000']
        dses = ['ChlorineConcentration', 'ElectricDevices', 'FordA', 'NonInvasiveFetalECGThorax1', 'TwoPatterns',
                'Wafer']
        # dses = ['ChlorineConcentration','ElectricDevices','EthanolLevel','FordA',
        #         'NonInvasiveFetalECGThorax1','Ozone','RefrigerationDevices','SemgHandSubjectCh2',
        #         'TwoPatterns','Wafer']
        ofses = ['-','Alpha Investing', 'Fast OSFS', 'FIRES', 'OSFS', 'SAOLA']
        oles = ['Neural Netwrok','K-Nearest Neighbors 3','K-Nearest Neighbors 5','Naive Bayes','Random Forest']
        base_path = r'C:\Users\Roi\Desktop\ans\{}\{}\{}\{}'
        window_path = r'C:\Users\Roi\Desktop\ans'
        for window in windows:
            run_times = {}
            for ol in oles:
                if ol not in run_times:
                    run_times[ol] = {}
                for ds in dses:
                    ofs_runtimes = []
                    for ofs in ofses:
                        path = os.path.join(base_path.format(ds,window,ofs,ol),'params.csv')
                        if not os.path.exists(path):
                            print(path)
                            continue
                        df = pd.read_csv(path)
                        ol_runtime = float(df.at[0, 'Mean OL algorithm runtime'])*1000
                        ofs_runtime = float(df.at[0, 'Mean OFS algorithm runtime'])*1000
                        if ofs == '-':
                            without_runtime = ol_runtime
                            continue
                        ofs_runtimes.append(ol_runtime+ofs_runtime)

                    run_times[ol][ds] = f'{np.mean(ofs_runtimes):.3f}, {without_runtime:.3f}'
            export_path = os.path.join(window_path,f"{window}_runtimes.csv")

            pd.DataFrame(run_times).to_csv(export_path)


    @staticmethod
    def get_ofs_times():
        windows = ['1000']
        dses = ['ChlorineConcentration', 'ElectricDevices', 'FordA', 'NonInvasiveFetalECGThorax1', 'TwoPatterns',
                'Wafer']
        ofses = [ 'SAOLA','Alpha Investing', 'OSFS','Fast OSFS', 'FIRES' ]
        oles = ['Neural Netwrok', 'K-Nearest Neighbors 3', 'K-Nearest Neighbors 5', 'Naive Bayes', 'Random Forest']
        base_path = r'C:\Users\Roi\Desktop\ans\{}\{}\{}\{}'
        window_path = r'C:\Users\Roi\Desktop\ans\{}'
        run_times = {}
        for ds in dses:
            for ofs in ofses:
                run_times[ofs] = {}
                ofs_runtimes = []
                for ol in oles:
                    path = os.path.join(base_path.format(ds, 1000, ofs, ol), 'params.csv')
                    if not os.path.exists(path):
                        print(path)
                        continue
                    df = pd.read_csv(path)
                    ofs_runtime = float(df.at[0, 'Mean OFS algorithm runtime']) * 1000
                    ofs_runtimes.append(ofs_runtime)

                run_times[ofs]['1000'] = f'{np.mean(ofs_runtimes):.3f}'
            export_path = os.path.join(window_path.format(ds), f"{1000}_ofs_runtimes.csv")

            pd.DataFrame(run_times).to_csv(export_path)

    @staticmethod
    def get_accuracy():
        dses = ['ChlorineConcentration', 'ElectricDevices', 'FordA', 'NonInvasiveFetalECGThorax1', 'TwoPatterns',
                'Wafer']
        ofses = [ 'Alpha Investing','Fast OSFS','OSFS','SAOLA','-' ]
        ol = 'K-Nearest Neighbors 3'
        base_path = r'C:\Users\Roi\Desktop\ans\{}\{}\{}\{}'
        window_path = r'C:\Users\Roi\Desktop\ans'
        acc = {}
        for ofs in ofses:
            acc[ofs] = {}
            for ds in dses:

                path = os.path.join(base_path.format(ds, 1000, ofs, ol), 'params.csv')
                df = pd.read_csv(path)
                last_acc = float(df.at[0, 'Last accuracy'])


                acc[ofs][ds] = f'{last_acc:.3f}'

        export_path = os.path.join(window_path, f"{ol}_1000.csv")
        pd.DataFrame(acc).to_csv(export_path)

    @staticmethod
    def rename_data():
        dses = ['ChlorineConcentration','ElectricDevices','FordA','NonInvasiveFetalECGThorax1','TwoPatterns','Wafer']
        ofses = ['-', 'Alpha Investing', 'Fast OSFS', 'FIRES', 'OSFS', 'SAOLA']
        base_path = r'C:\Users\Roi\Desktop\ans\{}\1000\{}'
        for ds in dses:
            for ofs in ofses:
                source = os.path.join(base_path.format(ds,ofs),'K-Nearest Neighbors')
                dest = os.path.join(base_path.format(ds, ofs), 'K-Nearest Neighbors 5')
                try:
                    shutil.copytree(source, dest)
                except Exception as e:
                    print('err')
    @staticmethod
    def copy_data():
        dses = ['ChlorineConcentration', 'ElectricDevices', 'FordA', 'NonInvasiveFetalECGThorax1', 'TwoPatterns',
                'Wafer']
        ofses = ['-', 'Alpha Investing', 'Fast OSFS', 'FIRES', 'OSFS', 'SAOLA']
        source_base_path = r'C:\Users\Roi\Desktop\knn3\{}\1000\{}\K-Nearest Neighbors 3'
        dest_base_path = r'C:\Users\Roi\Desktop\ans\{}\1000\{}\K-Nearest Neighbors 3'
        count = 0
        for ds in dses:
            for ofs in ofses:
                source = source_base_path.format(ds,ofs)
                dest = dest_base_path.format(ds, ofs)
                print(source)
                print(dest)
                try:
                    os.rename(source,dest)
                except Exception as e:
                    print(e)
                    print('err')

    @staticmethod
    def get_binary_distribution():
        file_paths = [
            r'C:\Users\Roi\Documents\Degree\Semester 8\פרוייקט גמר\datasets\new\ChlorineConcentration\ChlorineConcentration_TRAIN.arff',
            r'C:\Users\Roi\Documents\Degree\Semester 8\פרוייקט גמר\datasets\new\ElectricDevices\ElectricDevices_TRAIN.arff',
            r'C:\Users\Roi\Documents\Degree\Semester 8\פרוייקט גמר\datasets\new\EthanolLevel\EthanolLevel_TRAIN.arff',
            r'C:\Users\Roi\Documents\Degree\Semester 8\פרוייקט גמר\datasets\new\FordA\FordA_TRAIN.arff',
            r'C:\Users\Roi\Documents\Degree\Semester 8\פרוייקט גמר\datasets\new\NonInvasiveFetalECGThorax1\NonInvasiveFetalECGThorax1_TRAIN.arff',
            r'C:\Users\Roi\Documents\Degree\Semester 8\פרוייקט גמר\datasets\new\Ozone Level Detection Data Set\ozone.csv',
            r'C:\Users\Roi\Documents\Degree\Semester 8\פרוייקט גמר\datasets\new\RefrigerationDevices\RefrigerationDevices_TRAIN.arff',
            r'C:\Users\Roi\Documents\Degree\Semester 8\פרוייקט גמר\datasets\new\SemgHandSubjectCh2\SemgHandSubjectCh2_TRAIN.arff',
            r'C:\Users\Roi\Documents\Degree\Semester 8\פרוייקט גמר\datasets\new\TwoPatterns\TwoPatterns_TRAIN.arff',
            r'C:\Users\Roi\Documents\Degree\Semester 8\פרוייקט גמר\datasets\new\Wafer\Wafer_TRAIN.arff'
        ]
        file_names = ['ChlorineConcentration', 'ElectricDevices', 'EthanolLevel', 'FordA', 'NonInvasiveFetalECGThorax1',
                      'Ozone',
                      'RefrigerationDevices', 'SemgHandSubjectCh2',
                      'TwoPatterns', 'Wafer']
        export_path = r"C:\Users\Roi\Desktop\knn3_bin"
        headers = ['Dataset','Num of Records', 'Num of 0 records','Percentage of 0 records', 'Num of 1 records','Percentage of 1 records']
        data = []
        for file_name, file_path in zip(file_names,file_paths):
            X, y, classes = Parse.read_ds(file_path, target_index=-1)
            bin_y = Experiment.transform_binary(y)
            unique, counts = np.unique(bin_y, return_counts=True)
            print(unique,counts)
            data.append([file_name,bin_y.shape[0],counts[0],counts[0]/bin_y.shape[0],counts[1],counts[1]/bin_y.shape[0]])

        pd.DataFrame(data,columns=headers).to_csv(os.path.join(export_path,"bin_dist.csv"))
if __name__ == '__main__':
    General.get_binary_distribution()