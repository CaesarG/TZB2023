import timm
import torch.nn as nn
import torch
from model.efficientnet_pytorch import EfficientNet
import numpy as np
from dataset.dataset_ifft import myDataset_ifft
from torch.utils.data import DataLoader
import numpy as np
import sklearn.metrics

NUM_WORKERS = 24
NUM_CLASSES = 9


def accuracy(x, gt):
    predicted = np.argmin(x, axis=1)
    total = len(gt)
    acc = np.sum(predicted == gt) / total
    return acc


def acc_predicted(predicted, gt):
    total = len(gt)
    acc = np.sum(predicted == gt) / total
    return acc


def auroc(inData, outData, in_low=True):
    inDataMin = np.min(inData, 1)
    outDataMin = np.min(outData, 1)

    allData = np.concatenate((inDataMin, outDataMin))
    labels = np.concatenate((np.zeros(len(inDataMin)), np.ones(len(outDataMin))))
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, allData, pos_label=in_low)

    return sklearn.metrics.auc(fpr, tpr), thresholds


def get_model(cust=False):
    if cust:
        model = EfficientNet.from_pretrained(
            "efficientnet-b0", in_channels=2, num_classes=NUM_CLASSES
        )

    else:
        model = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=NUM_CLASSES
        )
        model.conv_stem = nn.Conv2d(
            2, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        # model.conv_stem=nn.Conv2d(2,32,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
    model.cuda()

    return model


class Openmax(nn.Module):
    def __init__(self, weight_path,base_path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = get_model(cust=False)
        self.num_classes = NUM_CLASSES
        self.anchors = nn.Parameter(
            torch.zeros(self.num_classes, self.num_classes).double()
        )
        self.load_weight(weight_loc=weight_path)
        self.set_dataloader(base_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_anchors(self, means):
        means = torch.Tensor(np.array(means))
        self.anchors = nn.Parameter(means.double())
        self.cuda()

    def distance_classifier(self, x):
        n = x.size(0)
        m = self.num_classes
        d = self.num_classes

        x = x.unsqueeze(1).expand(n, m, d).double()
        anchors = self.anchors.unsqueeze(0).expand(n, m, d).cuda()
        dists = torch.norm(x - anchors, 2, 2)
        return dists

    def get_x_dist(self, data):
        x = self.model(data)
        return x, self.distance_classifier(x)

    def load_weight(self, weight_loc):
        checkpoint = torch.load(weight_loc, map_location=lambda storage, loc: storage)
        state_dict = checkpoint["model_state_dict"]
        self.model.load_state_dict(state_dict)

    def set_dataloader(self, base_path, ifft=False):
        dataset = myDataset_ifft
        train_set = dataset(annotation_lines=base_path + "train.txt")
        val_set = dataset(annotation_lines=base_path + "val.txt")
        test_set = dataset(annotation_lines=base_path + "test.txt")
        openset_test = dataset(annotation_lines=base_path + "open_test.txt")
        openset_val = dataset(annotation_lines=base_path + "open_val.txt")

        # train_set=dataset(annotation_lines='dataRCS/annotations/train.txt')
        # val_set=dataset(annotation_lines='dataRCS/annotations/val.txt')

        self.dataloader_close_train = torch.utils.data.DataLoader(
            train_set,
            batch_size=1,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        self.dataloader_close_val = torch.utils.data.DataLoader(
            val_set,
            batch_size=1,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        self.dataloader_open_val = torch.utils.data.DataLoader(
            openset_val,
            batch_size=1,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        ...

    def get_output(
        self, dataloader, data_idx=0, calculate_scores=False, only_correct=False
    ):
        activation = []
        X = []
        y = []
        y_predict = []

        if calculate_scores:
            softmax = torch.nn.Softmax(dim=1)
        for batch_idx, (data, gt) in enumerate(dataloader):
            gt = gt.to(device=self.device, non_blocking=True)
            data = data.to(device=self.device, non_blocking=True)
            logits, distances = self.get_x_dist(data)
            _, predicted = torch.min(distances, dim=1)

            # Filter out wrong data use masks
            if only_correct:
                mask = predicted == gt
                logits = logits[mask]
                distances = distances[mask]
                gt = gt[mask]

            # Calculate scores (probability of rejection)
            if calculate_scores:
                softmin = softmax(-distances)
                invScores = 1 - softmin
                scores = distances * invScores
                _, predicted = torch.min(distances, dim=1)
            else:
                if data_idx == 0:
                    scores = logits
                if data_idx == 1:
                    scores = distances
            # Attach data to list
            activation += logits.cpu().detach().tolist()
            X += scores.cpu().detach().tolist()
            y += gt.cpu().tolist()
            y_predict += predicted.cpu().tolist()
        # Make it numpy array
        activation = np.asarray(activation)
        X = np.asarray(X)
        y = np.asarray(y)
        y_predict = np.asarray(y_predict)

        return activation, X, y, y_predict

    def find_anchor_means(self, only_correct=False):
        means = [None for i in range(self.num_classes)]

        activation, logits, labels, y_predict = self.get_output(
            self.dataloader_close_train, only_correct=only_correct
        )

        for cl in range(self.num_classes):
            x = logits[labels == cl]
            x = np.squeeze(x)
            means[cl] = np.mean(x, axis=0)

        return means

    def set_threshold(self, threshold):
        self.threshold = threshold

    def get_predicted(self, total_scores,print_opt):
        predicted = np.argmin(total_scores, axis=1)
        # reject the smaple with distance larger than threshold
        min_score = np.min(total_scores, axis=1)
        predicted[min_score > self.threshold] = 9
        if(print_opt):
            all_min = np.min(min_score)
            all_max = np.max(min_score)
            print("range of min_score is " + str(all_min) + " to " + str(all_max))
        return predicted


if __name__ == "__main__":
    alpha = 5
    num_classes = NUM_CLASSES
    anchors = torch.diag(torch.Tensor([alpha for i in range(num_classes)]))
    classifier = Openmax("weight_of_model/model_best_331.pth",base_path='openset_annotations/annotations1/')
    classifier.set_anchors(anchors)
    classifier.eval()
    classifier.set_anchors(classifier.find_anchor_means(only_correct=False))
    classifier.set_threshold(20)
    (
        total_activation_know,
        total_scores_know,
        total_gt_know,
        total_predicts_know,
    ) = classifier.get_output(
        classifier.dataloader_close_val,
        data_idx=1,
        calculate_scores=True,
        only_correct=False,
    )
    total_activation, total_scores, total_gt, total_predicts = classifier.get_output(
        classifier.dataloader_open_val,
        data_idx=1,
        calculate_scores=True,
        only_correct=False,
    )
    open_predicted = classifier.get_predicted(total_scores,True)
    accuracy_know = accuracy(total_scores_know, total_gt_know)
    accuracy_open = acc_predicted(open_predicted, total_gt)
    # auroc, th = auroc(total_scores_know[:, 1:], total_scores[:, 1:])
    # print(auroc.shape)
    # print(th.shape)
    print("acc_know=", accuracy_know)
    print("acc_open=", accuracy_open)
    auroc,th = auroc(total_scores_know, total_scores)
    print('AUROC =', auroc)
    # print('AUROC_TH =', th.shape)
    # Test threshold
    for test_th in th:
        classifier.set_threshold(test_th)
        test_predicted=classifier.get_predicted(total_scores,False)
        accuracy_open = acc_predicted(test_predicted, total_gt)
        print(accuracy_open)
    
