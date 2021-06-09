from Model.Simulation.experiment import Experiment
from Model.OFS.ofs_ac import OnlineFeatureSelectionAC
from Model.OL.ol_ac import OnlineLearningAC
from Model.Simulation.parse import Parse


ofs_algos = OnlineFeatureSelectionAC.get_all_ofs_algo()
ol_models = OnlineLearningAC.get_all_ol_algo()

def single_experiment_test(file_path,file_name,file_target_index=-1,window_size=300,ol_index=0,ofs_index=0):
    X, y, classes = Parse.read_ds(file_path, target_index=file_target_index)
    ofs_instance, ol_instance = ofs_algos[ofs_index](), ol_models[ol_index]()
    experiment = Experiment(ofs=ofs_instance, ol=ol_instance, window_size=window_size, X=X, y=y, ds_name=file_name,
                            transform_binary=False, special_name='single_test')
    experiment.ol.set_algorithm_fit_parameters(classes=classes)
    try:
        experiment.run()
        experiment.save()
    except Exception as e:
        print(experiment)
        print(e)


def multi_experiments_test(file_path, file_name, export_path,file_target_index=-1,window_sizes=[300, 500]):

    X, y, classes = Parse.read_ds(file_path, target_index=file_target_index)
    ds_exps = []
    ofs_algos["without"] = None
    for window_size in window_sizes:
        for ofs_name, ofs_class in ofs_algos.items():
            if ofs_name != 'without':
                continue
            ofs_instance = ofs_class() if ofs_class else ofs_class
            for ol_name, ol_class in ol_models.items():
                if ol_name!= 'KNN':
                    continue
                ol_instance = ol_class()
                ol_instance.set_algorithm_fit_parameters(classes=classes)
                experiment = Experiment(ofs=ofs_instance, ol=ol_instance, window_size=window_size, X=X, y=y,
                                        ds_name=file_name, transform_binary=True, special_name='multi')
                ds_exps.append(experiment)
                experiment.run()
                experiment.save(path=export_path)
            # window_instance.increase_pb()
    Experiment.save_graphs(ds_exps)


if __name__ == '__main__':
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
    file_names =  ['ChlorineConcentration','ElectricDevices','EthanolLevel','FordA','NonInvasiveFetalECGThorax1','Ozone',
                   'RefrigerationDevices','SemgHandSubjectCh2',
                'TwoPatterns','Wafer']
    export_path = r"C:\Users\Roi\Desktop\knn3_bin"
    # single_experiment_test(file_path,file_name, file_target_index=-1, window_size=300, ol_index=0, ofs_index=0)
    # multi_experiments_test(file_path, file_name, file_target_index=-1, window_sizes=[1000])
    for file_name,file_path in zip(file_names,file_paths):
        multi_experiments_test(file_path, file_name, export_path,file_target_index=-1,
                          window_sizes=[300,500])

