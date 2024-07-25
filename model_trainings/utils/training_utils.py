from torch.utils.data import Dataset
import torchvision
import torch
from torch import nn, optim

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, balanced_accuracy_score

import numpy as np

import matplotlib.pyplot as plt

class Triangles(Dataset):
    def __init__(self,imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img=self.imgs[idx]
        label=self.labels[idx]
        
        sample = {'image': img, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
def build_batches(x, n):
    x = np.asarray(x)
    m = (x.shape[0] // n) * n
    return x[:m].reshape(-1, n, *x.shape[1:])


class LeNet5(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(7744, 120) #assumes input of 100x100 images
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
     

def get_net_optimiser_scheduler_criterion(device,class_weights=None, model_type = 'resnet18'):
    if model_type == 'resnet18':
        net = torchvision.models.resnet18(pretrained=False)
        #net = torchvision.models.resnet34(pretrained=False)
        net.conv1= nn.Conv2d(2, net.conv1.out_channels, 
                                    net.conv1.kernel_size,  
                                    net.conv1.stride,
                                    net.conv1.padding,
                                    bias=net.conv1.bias)  

        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 2)
    elif model_type == 'lenet':
        net = LeNet5()
    elif model_type == 'squeezenet':
        net = torchvision.models.squeezenet1_1(pretrained=True)
        # Get the number of input channels (3 for RGB) and output channels (e.g., 64)
        in_channels = net.features[0].in_channels
        out_channels = net.features[0].out_channels

        # Create a new Conv2d layer with 2 input channels instead of 3
        new_first_layer = nn.Conv2d(2, out_channels, kernel_size=3, stride=2)

        # Replace the first layer in the model's 'features' module
        net.features[0] = new_first_layer
        
        # Get the number of input features for the final layer
        in_features = net.classifier[1].in_channels
        out_features = net.classifier[1].out_channels

        # Create a new Conv2d layer with 2 output channels (for 2 classes) instead of the original number (e.g., 1000)
        new_final_layer = nn.Conv2d(in_features, 2, kernel_size=1)

        # Replace the final layer in the model's 'classifier' module
        net.classifier[1] = new_final_layer
        
        net.num_classes = 2
    else:
        raise(f'model_type {model_type} not considered')
    net = net.to(device)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    return net, optimizer, scheduler, criterion

def get_results_dict(include_latest_basel_experiment=False):
    
    d={'confusion matrix':[],
                 'TPR':[], 'TNR':[], 'Accuracy':[], 'Balanced Accuracy':[],
                  'AUC':[], 'ROC Curve':[],
                  'true_labels':[], 'predictions':[], 'scores':[], 'triangle_names':[],
               'device_names':[],
                 }
    if include_latest_basel_experiment:
        d['latest Basel']={}
        d['latest Basel']['Accuracy']=[]
        d['latest Basel']['TPR']=[]
        d['latest Basel']['TNR']=[]
        d['latest Basel']['confusion matrix']=[]
        d['latest Basel']['Balanced Accuracy']=[]
        
    return d

def record_results(results, predicted, scores, true_labels, device_names=None, triangle_names=None,mode='classic',
                   latest_basel_idx=None, include_latest_basel_experiment=False):
    if mode=='classic':
        results['triangle_names'].append(triangle_names)
        results['device_names'].append(device_names)
        
        cm=confusion_matrix(true_labels,predicted, labels=[0,1])

        results['confusion matrix'].append(cm)
        tn, fp, fn, tp = cm.ravel()

        tnr = tn / (tn+fp)
        tpr = tp / (tp+fn)
        results['Accuracy'].append((tp+tn)/(tn+ fp+ fn+ tp))
        results['Balanced Accuracy'].append(balanced_accuracy_score(true_labels,predicted))
        results['TPR'].append(tpr)
        results['TNR'].append(tnr)

       

        fpr, tpr, thresholds = roc_curve(true_labels, scores)

        results['scores'].append(scores)
        results['ROC Curve'].append([fpr,tpr])

        AUC=roc_auc_score(true_labels, scores)

        results['AUC'].append(AUC)

        results['predictions'].append(predicted)
        results['true_labels'].append(true_labels)

        if include_latest_basel_experiment:
            cm=confusion_matrix(true_labels[latest_basel_idx],predicted[latest_basel_idx], labels=[0,1])
            tn, fp, fn, tp = cm.ravel()
            tnr = tn / (tn+fp)
            tpr = tp / (tp+fn)
            results['latest Basel']['Accuracy'].append((tp+tn)/(tn+ fp+ fn+ tp))
            results['latest Basel']['Balanced Accuracy'].append(balanced_accuracy_score(true_labels[latest_basel_idx]
                                                                                        ,predicted[latest_basel_idx]))
            results['latest Basel']['TPR'].append(tpr)
            results['latest Basel']['TNR'].append(tnr)
            results['latest Basel']['confusion matrix'].append(cm)
        return results
    else:
        results['triangle_names'].append(triangle_names)
        results['predictions'].append(predicted)
        results['scores'].append(scores)
        results['device_names'].append(device_names)
        results['true_labels'].append(true_labels)
        
        #weigh each device equally
        test_devices=['Tuor6A_chiplet_5_device_C',
           'Tuor6A_chiplet_6_device_E', 'Tuor6A_chiplet_7_device_A']
        
        
        _tnr=[]
        _tpr=[]
        _accuracy=[]
        _balanced_accuracy=[]
        _roc_curve=[]
        _auc=[]
        _cms={}
        
        for test_device_name in test_devices:
            print('testing',test_device_name )
            if test_device_name=='Tuor6A_chiplet_5_device_C':
                test_index =np.logical_or(device_names=='Tuor6A_chiplet_5_device_C_cooldown_1' ,
                              device_names=='Tuor6A_chiplet_5_device_C_cooldown_2')
            else:
                test_index =device_names==test_device_name
            
            cm=confusion_matrix(true_labels[test_index],predicted[test_index], labels=[0,1])

            _cms[test_device_name]=cm
            tn, fp, fn, tp = cm.ravel()
            _accuracy.append((tp+tn)/(tn+ fp+ fn+ tp))
            _balanced_accuracy.append(balanced_accuracy_score(true_labels[test_index],predicted[test_index]))
            
            _tnr.append(tn / (tn+fp))
            _tpr.append(tp / (tp+fn))
            


            _roc_curve.append(roc_curve(true_labels[test_index], scores[test_index]))
            #fpr, tpr, thresholds
            
            

            _auc.append(roc_auc_score(true_labels[test_index], scores[test_index]))

            
        results['confusion matrix'].append(_cms)
        results['Accuracy'].append(np.mean(_accuracy))
        results['Balanced Accuracy'].append(np.mean(_balanced_accuracy))
        results['TPR'].append(np.mean(_tpr))
        results['TNR'].append(np.mean(_tnr))
        results['AUC'].append(np.mean(_auc))
        
        tprs = []
        #aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        for i, (fpr, tpr, thresholds) in enumerate(_roc_curve):
            #mean_fpr = np.linspace(0, 1, 1000)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        results['ROC Curve'].append([mean_fpr,mean_tpr])
        
        return results

def report_results(results, include_latest_basel_experiment=False):
    print('Confusion matrix: ',results['confusion matrix'][-1])
    print('TPR and TNR:',results['TPR'][-1], results['TNR'][-1])
    print('Accuracy:',results['Accuracy'][-1])
    print('AUC:',results['AUC'][-1])
    print('')
    print('Mean/Std AUC', np.mean(results['AUC']), np.std(results['AUC']))
    print('Mean/Std accuracy', np.mean(results['Accuracy']), np.std(results['Accuracy']))
    print('Mean/Std balanced accuracy', np.mean(results['Balanced Accuracy']), np.std(results['Balanced Accuracy']))
    
    
    plt.plot(results['Accuracy'], label='Accuracy', color='tab:red')
    plt.plot(results['Balanced Accuracy'], label='Balanced Accuracy', color='tab:pink')
    plt.plot(results['TPR'], label='TPR', color='tab:green')
    plt.plot(results['TNR'], label='TNR', color='tab:blue')

    plt.plot(results['AUC'], label='AUC', color='tab:orange', linestyle='-.')
    if include_latest_basel_experiment:
        plt.plot(results['latest Basel']['TPR'], label='TPR latest', color='tab:green', linestyle='--')
        plt.plot(results['latest Basel']['TNR'], label='TNR latest', color='tab:blue', linestyle='--')
        print('')
        print('Only latest Basel experiment:')
        print('Confusion matrix: ',results['latest Basel']['confusion matrix'][-1])
        print('TPR and TNR:',results['latest Basel']['TPR'][-1], results['latest Basel']['TNR'][-1])
        print('Accuracy:',results['latest Basel']['Accuracy'][-1])
        print('')
        print('Mean/Std TPR', np.mean(results['latest Basel']['TPR']), np.std(results['TPR']))
        print('Mean/Std TNR', np.mean(results['latest Basel']['TNR']), np.std(results['TNR']))
        print('Mean/Std accuracy', np.mean(results['latest Basel']['Accuracy']), np.std(results['latest Basel']['Accuracy']))
        print('Mean/Std balanced accuracy', 
              np.mean(results['latest Basel']['Balanced Accuracy']),
              np.std(results['latest Basel']['Balanced Accuracy']))

    plt.legend()
    plt.show()

    
    [fpr,tpr] = results['ROC Curve'][-1]
    plt.plot(fpr,tpr,color='tab:green', label='latest run')
    
    if len(results['ROC Curve'])>1:
        for [fpr,tpr] in results['ROC Curve'][:-1]:
            plt.plot(fpr,tpr,color='tab:grey', alpha=0.1)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.show()