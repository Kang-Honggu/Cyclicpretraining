import os
import sys
import shutil
import time
import numpy as np
from optparse import OptionParser
from shutil import copyfile
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from optparse import OptionParser
from sklearn.metrics import roc_auc_score
from utils import vararg_callback_bool, vararg_callback_int, get_config, metric_AUROC, cosine_scheduler
from dataloader import  *
from model import *
from trainer import *


from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma


import torch
#from engine import ark_engine

sys.setrecursionlimit(40000)

def main():
    datasets_config = get_config('datasets_config.yaml')
    datasets = list(datasets_config.keys())
    
    dataset_train_list = []
    dataset_val_list = []
    dataset_test_list = []
    for dataset in datasets:
        dataset_train_list.append(
            Eye_Dataset(images_path=datasets_config[dataset]['data_dir'], file_path=datasets_config[dataset]['train_list'],
                                 imagetype=datasets_config[dataset]['imagetype'],train=True)
        )
        dataset_val_list.append(
            Eye_Dataset(images_path=datasets_config[dataset]['data_dir'], file_path=datasets_config[dataset]['val_list'],
                                 imagetype=datasets_config[dataset]['imagetype'],train=False)
        )
        dataset_test_list.append(
            Eye_Dataset(images_path=datasets_config[dataset]['data_dir'], file_path=datasets_config[dataset]['test_list'],
                                 imagetype=datasets_config[dataset]['imagetype'], train=False)
        )
    
    data_loader_list_train = []
    for d in dataset_train_list:
        data_loader_list_train.append(DataLoader(dataset=d, batch_size=64, shuffle=True,
                                        num_workers=4, pin_memory=True))
    data_loader_list_val = []
    for dv in dataset_val_list:
        data_loader_list_val.append(DataLoader(dataset=dv, batch_size=64, shuffle=False,
                                        num_workers=4, pin_memory=True))
    data_loader_list_test = []
    for dt in dataset_test_list: 
        data_loader_list_test.append(DataLoader(dataset=dt, batch_size=32, shuffle=False,
                                        num_workers=4, pin_memory=True))
    #ark_engine(args, model_path, output_path, args.dataset_list, datasets_config, dataset_train_list, dataset_val_list, dataset_test_list)

    print("Datasets \n",datasets)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device : ",device)
    print('Name of device :',torch.cuda.get_device_name())
    num_classes_list = [len(datasets_config[dataset]['diseases']) for dataset in datasets]
    print('num_classes_list : ',num_classes_list)
    criterion = torch.nn.BCEWithLogitsLoss()

    model = Resnet50_ark(num_classes_list)
    teacher_model = Resnet50_ark(num_classes_list)

    model.to(device)
    teacher_model.to(device)
    
    for p in teacher_model.parameters():
        p.requires_grad = False

    #momentum_schedule = cosine_scheduler(0.9,1,10,len(datasets))  #코드 보면서 이해가 필요. 일단 default 값 적용
    #coef_schedule = cosine_scheduler(0,0.5,10,len(datasets))
        """
    class Args:
        opt = 'adam'  # 사용할 optimizer 이름 (예: sgd, adam, adamw 등)
        lr = 1e-4  # 학습률
        weight_decay = 1e-2  # weight decay (L2 regularization)
        opt_eps = 1e-8  # epsilon (Adam의 경우)
        opt_betas = (0.9, 0.999)  # betas (Adam의 경우)
        momentum = 0.9
        
        sched = 'cosine'
        epochs = 30
        min_lr = 1e-5
        warmup_epochs = 5
        warmup_lr = 1e-6
        decay_rate = 0.1
        decay_epochs = 30

    args = Args()
    """
    #optimizer = create_optimizer(args=args,model= model)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)
    #lr_scheduler, _ = create_scheduler(optimizer= optimizer, args=args)

    start_epoch = 0
    init_loss = 999999
    best_val_loss = init_loss
    save_model_path = os.path.join("./Models","Adamw with cosine")
    output_path = os.path.join("./Outputs")

    test_results,test_results_teacher = [],[]
    
    criterion = torch.nn.BCEWithLogitsLoss()
    for epoch in range(start_epoch, 30):
        for i, data_loader in enumerate(data_loader_list_train): 
            print("Train with ",datasets[i])
            train_one_epoch(model, i, datasets[i], data_loader, device, criterion, optimizer, epoch, teacher_model)
        val_loss_list = []
        for i, dv in enumerate(data_loader_list_val):
            val_loss = evaluate(model, i, dv, device, criterion, datasets[i])
            val_loss_list.append(val_loss)
            # wandb.log({"val_loss_{}".format(dataset_list[i]): val_loss})
        
        avg_val_loss = np.average(val_loss_list)
        #if args.val_loss_metric == "average":  => early stopping을 위한 threslhold
        val_loss_metric = avg_val_loss
        """
        else:
            val_loss_metric = val_loss_list[dataset_list.index(args.val_loss_metric)]
        lr_scheduler.step(val_loss_metric)
        """
        # log metrics to wandb
        # wandb.log({"avg_val_loss": avg_val_loss})
        scheduler1.step()
        print("Epoch {:04d}: avg_val_loss {:.5f}, saving model to {}".format(epoch, avg_val_loss,save_model_path))
        save_checkpoint({
                'epoch': epoch,
                'lossMIN': val_loss_list,
                'state_dict': model.state_dict(),
                'teacher': teacher_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler1.state_dict(),
                },  filename=save_model_path)

        log_file = os.path.join("./Models", "train.log")

        with open(log_file, 'a') as log:
            log.write("Epoch {:04d}: avg_val_loss = {:.5f} \n".format(epoch, avg_val_loss))
            log.write("     Datasets  : " + str(datasets) + "\n")
            log.write("     Val Losses: " + str(val_loss_list) + "\n")
            log.close()

        #if epoch % args.test_epoch == 0 or epoch+1 == 10:
        if (epoch+1)%10 == 0:          
            save_checkpoint({
                    'epoch': epoch,
                    'lossMIN': val_loss_list,
                    'state_dict': model.state_dict(),
                    'teacher': teacher_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler1.state_dict(),
                    },  filename=save_model_path+str(epoch))
            output_file = os.path.join(output_path, "_results.txt")
            with open(output_file, 'a') as writer:
                writer.write("Omni-pretraining stage:\n")
                writer.write("Epoch {:04d}:\n".format(epoch))
                t_res, t_res_teacher = [],[]
                for i, dataset in enumerate(datasets):
                    writer.write("{} Validation Loss = {:.5f}:\n".format(dataset, val_loss_list[i]))
                    diseases = datasets_config[dataset]['diseases']
                    print(">>{} Disease = {}".format(dataset, diseases))
                    writer.write("{} Disease = {}\n".format(dataset, diseases))

                    multiclass =  datasets_config[dataset]['task_type'] == "multi-class classification"
                    y_test, p_test = test_classification(model, i, data_loader_list_test[i], device, multiclass)
                    y_test_teacher, p_test_teacher = test_classification(teacher_model, i, data_loader_list_test[i], device, multiclass)
                    if multiclass:
                        acc = accuracy_score(np.argmax(y_test.cpu().numpy(),axis=1),np.argmax(p_test.cpu().numpy(),axis=1))
                        acc_teacher = accuracy_score(np.argmax(y_test_teacher.cpu().numpy(),axis=1),np.argmax(p_test_teacher.cpu().numpy(),axis=1))
                        print(">>{}:Student ACCURACY = {}, \nTeacher ACCURACY = {}\n".format(dataset,acc, acc_teacher))
                        writer.write(
                            "\n{}: Student ACCURACY = {}, \nTeacher ACCURACY = {}\n".format(dataset, np.array2string(np.array(acc), precision=4, separator='\t'), np.array2string(np.array(acc_teacher), precision=4, separator='\t')))   
                        t_res.append(acc)
                        t_res_teacher.append(acc_teacher)

                    print("y test : ",y_test)
                    print("p_test : ",p_test)
                    y_test = y_test.cpu().numpy()
                    p_test = p_test.cpu().numpy()
                    y_test_teacher = y_test_teacher.cpu().numpy()
                    p_test_teacher = p_test_teacher.cpu().numpy()
                    student_auc = metric_AUROC(y_test,p_test,num_classes_list[i])
                    teacher_auc = metric_AUROC(y_test_teacher,p_test_teacher,num_classes_list[i])
                    """
                    individual_results = metric_AUROC(y_test, p_test, len(diseases))
                    individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(diseases)) 
                    print(">>{}:Student AUC = {}, \nTeacher AUC = {}\n".format(dataset, np.array2string(np.array(individual_results), precision=4, separator='\t'),np.array2string(np.array(individual_results_teacher), precision=4, separator='\t')))
                    """
                    #(">>{}: Student mAUC = {:.4f}, Teacher mAUC = {:.4f}".format(dataset, student_auc,teachaer_auc))

                    print("dataset : ",dataset,"student_auc : ", student_auc, "teacher_auc : " , teacher_auc, "\n")
                    print("dataset : ",dataset,"\n Mean Student_AUC : ", np.mean(student_auc), "\n Mean Teacher_AUC : " , np.mean(teacher_auc), "\n")
                    writer.write(
                        "\n{}: Student AUC = {}, \nTeacher AUC = {}\n".format(dataset, np.mean(student_auc), np.mean(teacher_auc)))
                    
                    #mean_over_all_classes = np.array(individual_results).mean()
                    #mean_over_all_classes_teacher = np.array(individual_results_teacher).mean()
                    #print(">>{}: Student mAUC = {:.4f}, Teacher mAUC = {:.4f}".format(dataset, mean_over_all_classes,mean_over_all_classes_teacher))
                    #writer.write("{}: Student mAUC = {:.4f}, Teacher mAUC = {:.4f}\n".format(dataset, mean_over_all_classes,mean_over_all_classes_teacher))
                    #t_res.append(mean_over_all_classes)
                    #t_res_teacher.append(mean_over_all_classes_teacher)
                    
                writer.close()

                test_results.append(t_res)
                test_results_teacher.append(t_res_teacher)
    
                print("Omni-pretraining stage: \nStudent meanAUC = \n{} \nTeacher meanAUC = \n{}\n".format(test_results, test_results_teacher))
    with open(output_file, 'a') as writer:
        writer.write("Omni-pretraining stage: \nStudent meanAUC = \n{} \nTeacher meanAUC = \n{}\n".format(np.array2string(np.array(test_results), precision=4, separator='\t'),np.array2string(np.array(test_results_teacher), precision=4, separator='\t')))
    writer.close()



if __name__ == '__main__':
    main()

