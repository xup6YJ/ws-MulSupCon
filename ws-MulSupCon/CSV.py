import pandas as pd
import shutil
import os

def CreateCSVFile(disease_labels, csv_path = None):
    disease_columns = disease_labels
    if not os.path.exists("result"):
        os.makedirs("result")

    df = pd.DataFrame(columns=disease_columns)
    # df.to_csv(csv_path + "/train_single_acc.csv", index=False)
    df.to_csv(csv_path + "/test_single_acc.csv", index=False)
    # df.to_csv(csv_path + "/train_single_f1.csv", index=False)
    df.to_csv(csv_path + "/test_single_f1.csv", index=False)
    # df.to_csv(csv_path + "/train_single_recall.csv", index=False)
    df.to_csv(csv_path + "/test_single_recall.csv", index=False)
    # df.to_csv(csv_path + "/train_single_precision.csv", index=False)
    df.to_csv(csv_path + "/test_single_precision.csv", index=False)
    # df.to_csv(csv_path + "/train_AUC.csv", index=False)
    df.to_csv(csv_path + "/test_AUC.csv", index=False)

    columns = ["acc", "f1", "recall", "precision", "mAUC",   "train_loss"]
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_path + "/train_all.csv", index=False)

    columns = ["acc", "f1", "recall", "precision", "mAUC", "test_loss"]
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_path + "/test_all.csv", index=False)

    # columns = ["tp", "tn", "fp", "fn"] + disease_columns
    # df = pd.DataFrame(columns=columns)
    # df.to_csv(csv_path + "/train_true&false.csv", index=False)
    # df.to_csv(csv_path + "/test_true&false.csv", index=False)


def MoveCSVFile(move_path):

    # shutil.move("train_single_acc.csv", f"{move_path}/train_single_acc.csv")
    # shutil.move("test_single_acc.csv", f"{move_path}/test_single_acc.csv")
    # shutil.move("train_single_f1.csv", f"{move_path}/train_single_f1.csv")
    # shutil.move("test_single_f1.csv", f"{move_path}/test_single_f1.csv")
    # shutil.move("train_single_recall.csv", f"{move_path}/train_single_recall.csv")
    # shutil.move("test_single_recall.csv", f"{move_path}/test_single_recall.csv")
    # shutil.move("train_single_precision.csv", f"{move_path}/train_single_precision.csv")
    # shutil.move("test_single_precision.csv", f"{move_path}/test_single_precision.csv")
    # shutil.move("train_AUC.csv", f"{move_path}/train_AUC.csv")
    # shutil.move("test_AUC.csv", f"{move_path}/test_AUC.csv")
    # shutil.move("train_all.csv", f"{move_path}/train_all.csv")
    # shutil.move("test_all.csv", f"{move_path}/test_all.csv")
    # shutil.move("train_true&false.csv", f"{move_path}/train_true&false.csv")
    # shutil.move("test_true&false.csv", f"{move_path}/test_true&false.csv")
    # shutil.move("best_f1_epoch20.pt", f"{move_path}/best_f1_epoch20.pt")
    # shutil.move("best_mAUC_epoch20.pt", f"{move_path}/best_mAUC_epoch20.pt")
    # shutil.move("best_f1_epoch40.pt", f"{move_path}/best_f1_epoch40.pt")
    # shutil.move("best_mAUC_epoch40.pt", f"{move_path}/best_mAUC_epoch40.pt")
    
    os.system(f"mv result {move_path}/result")
