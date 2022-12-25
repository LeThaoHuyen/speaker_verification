# %% [markdown]
# # Config

# %%
# config
config = { 
    "dataroot": "./data",
    "n_triplets": 100000, #1000000,
    "embedding_size": 512,
    "batch_size": 512,
    "test_batch_size":  64,
    "lr": 0.1,
    "optimizer": "adagrad",
    "wd": 0.0,
    "lr_decay": 1e-4,
    "cuda": True,
    "start_epoch": 1,
    "epochs": 1,
    "min_softmax_epoch": 2,
    "margin": 0.1,
    "log_interval": 1,
    "loss_ratio": 2.0, 
    "test_input_per_file": 8,
    "test_batch_size": 64,
}

from easydict import EasyDict
config = EasyDict(config)

# %% [markdown]
# # DeepSpeaker model
# 

# %%
import torch
import torch.nn as nn
import math

from torch.autograd import Function


class PairwiseDistance(Function):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)

class TripletMarginLoss(Function):
    """Triplet loss function.
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)  # norm 2

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive)
        d_n = self.pdist.forward(anchor, negative)

        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss


class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class myResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):

        super(myResNet, self).__init__()

        self.relu = ReLU(inplace=True)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.inplanes = 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2,bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer2 = self._make_layer(block, 128, layers[1])

        self.inplanes = 256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2,bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.layer3 = self._make_layer(block, 256, layers[2])
        
        self.inplanes = 512
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2,bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.layer4 = self._make_layer(block, 512, layers[3])

        
        # self.avgpool = nn.AdaptiveAvgPool2d((1,None))
        self.avgpool = nn.AdaptiveAvgPool2d((None,1))
    
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)

    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)

    #     x = self.avgpool(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc(x)

    #     return x

class DeepSpeakerModel(nn.Module):
    def __init__(self,embedding_size,num_classes,feature_dim = 64):
        super(DeepSpeakerModel, self).__init__()

        self.embedding_size = embedding_size




        self.model = myResNet(BasicBlock, [1, 1, 1, 1])
        if feature_dim == 64:
            self.model.fc = nn.Linear(512*4, self.embedding_size)
            print("Hi I'm tired")
        elif feature_dim == 40:
            self.model.fc = nn.Linear(256 * 5, self.embedding_size)
        self.model.classifier = nn.Linear(self.embedding_size, num_classes)




    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)

        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer2(x)

        x = self.model.conv3(x)
        x = self.model.bn3(x)
        x = self.model.relu(x)
        x = self.model.layer3(x)

        x = self.model.conv4(x)
        x = self.model.bn4(x)
        x = self.model.relu(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)

        # print(x.size())
        x = self.model.fc(x)
       
        self.features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha=10
        self.features = self.features*alpha

        #x = x.resize(int(x.size(0) / 17),17 , 512)
        #self.features =torch.mean(x,dim=1)
        #x = self.model.classifier(self.features)
        return self.features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return res




# %% [markdown]
# # Data Loading

# %% [markdown]
# ## Train data

# %%
import pandas as pd

train_csv = pd.read_csv("./data/train_data.csv")
train_csv = train_csv[train_csv["is_audio"] == True]

# %%
train_csv

# %%
train_data = dict()

for i, row in train_csv.iterrows():
    speaker_id = row["speaker_id"]
    if speaker_id not in train_data:
        train_data[speaker_id] = []
    filepath = row["path_from_data_dir"]
    if ".wav" in filepath:
        train_data[speaker_id].append(filepath)
    

# %%
len(train_data.keys()) # number of speakers

# %% [markdown]
# ## Test data

# %%
import os
test_csv = pd.read_csv("./vox1_test_wav/test.csv")

# %%
test_pairs = []
valid_pairs = 0
for _, row in test_csv.iterrows():
    path1 = "./vox1_test_wav/wav/" + row["audio_1"]
    path2 = "./vox1_test_wav/wav/" + row["audio_2"]
    issame = True if row["label"] == '1' else False
    if os.path.exists(path1) and os.path.exists(path2):
        test_pairs.append((path1, path2, issame))
        valid_pairs += 1

valid_pairs

# %% [markdown]
# # Dataset transformation for training and testing

# %%
import numpy as np
import torch.utils.data as data

def generate_triplets(imgs, num_triplets,n_classes):
    def create_indices(_imgs):
        inds = dict()
        for idx, (img_path,label) in enumerate(_imgs):
            if label not in inds:
                inds[label] = []
            inds[label].append(img_path)
        return inds

    triplets = []
    # Indices = array of labels and each label is an array of indices
    indices = create_indices(imgs)

    #for x in tqdm(range(num_triplets)):
    for x in range(num_triplets):
        c1 = np.random.randint(0, n_classes)
        c2 = np.random.randint(0, n_classes)
        while len(indices[c1]) < 2:
            c1 = np.random.randint(0, n_classes)

        while c1 == c2:
            c2 = np.random.randint(0, n_classes)
        if len(indices[c1]) == 2:  # hack to speed up process
            n1, n2 = 0, 1
        else:
            n1 = np.random.randint(0, len(indices[c1]) - 1)
            n2 = np.random.randint(0, len(indices[c1]) - 1)
            while n1 == n2:
                n2 = np.random.randint(0, len(indices[c1]) - 1)
        if len(indices[c2]) ==1:
            n3 = 0
        else:
            n3 = np.random.randint(0, len(indices[c2]) - 1)

        triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3],c1,c2])
    return triplets

def find_id(data):
    speakerids = list(data.keys())
    speakerids.sort()
    speakerid2id = {speakerids[i]: i for i in range(len(speakerids))}
    return speakerids, speakerid2id

class DeepSpeakerDataset(data.Dataset):
    def __init__(self, data, n_triplets, loader, transform=None):
        self.classes, speakerid2id = find_id(data)
        imgs = []
        for speaker_id in data.keys():
            true_id = speakerid2id[speaker_id]
            for item in data[speaker_id]:
                imgs.append((config.dataroot + "/" + item, true_id))

        self.imgs = imgs
        self.loader = loader
        self.transform = transform
        self.n_triplets = n_triplets

        print('Generating {} triplets'.format(self.n_triplets))
        self.training_triplets = generate_triplets(self.imgs, self.n_triplets,len(self.classes))
    
    def __getitem__(self, index):
        '''
        Args:
            index: Index of the triplet or the matches - not of a single image

        Returns:

        '''
        def transform(img_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """

            img = self.loader(img_path)
            # img = np.load(img_path.replace('.wav', '.npy'))
            return self.transform(img)

        # Get the index of each image in the triplet
        a, p, n,c1,c2 = self.training_triplets[index]

        # transform images if required
        img_a, img_p, img_n = transform(a), transform(p), transform(n)
        return img_a, img_p, img_n, c1, c2

    def __len__(self):
        return len(self.training_triplets)
        

# %%
import os

class TestDataset(data.Dataset):
    def __init__(self, pairs, loader, transform=None):
        self.pairs = pairs
        self.loader = loader
        self.transform = transform
        
    def __getitem__(self, index):
        def transform(img_path):
            img = self.loader(img_path)
            return self.transform(img)       

        path1, path2, issame = self.pairs[index]
        img1, img2 = self.loader(path1), self.loader(path2)
        img1, img2 = transform(path1), transform(path2)
        return img1, img2, issame

    def __len__(self):
        return len(self.pairs)    

# %% [markdown]
# # Audio processing

# %%
NUM_PREVIOUS_FRAME = 9
#NUM_PREVIOUS_FRAME = 13
NUM_NEXT_FRAME = 23

NUM_FRAMES = NUM_PREVIOUS_FRAME + NUM_NEXT_FRAME
USE_LOGSCALE = True
USE_DELTA = False
USE_SCALE = False
SAMPLE_RATE = 16000
TRUNCATE_SOUND_FIRST_SECONDS = 0.5
FILTER_BANK = 64

# %%
import numpy as np
from python_speech_features import fbank, delta

import librosa

def normalize_frames(m,Scale=True):
    if Scale:
        return (m - np.mean(m, axis=0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return (m - np.mean(m, axis=0))

def mk_MFB(filename, sample_rate=SAMPLE_RATE,use_delta = USE_DELTA,use_scale = USE_SCALE,use_logscale = USE_LOGSCALE):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    #audio = audio.flatten()


    filter_banks, energies = fbank(audio, samplerate=sample_rate, nfilt=FILTER_BANK, winlen=0.025)

    if use_logscale:
        filter_banks = 20 * np.log10(np.maximum(filter_banks,1e-5))

    if use_delta:
        delta_1 = delta(filter_banks, N=1)
        delta_2 = delta(delta_1, N=1)

        filter_banks = normalize_frames(filter_banks, Scale=use_scale)
        delta_1 = normalize_frames(delta_1, Scale=use_scale)
        delta_2 = normalize_frames(delta_2, Scale=use_scale)

        frames_features = np.hstack([filter_banks, delta_1, delta_2])
    else:
        filter_banks = normalize_frames(filter_banks, Scale=use_scale)
        frames_features = filter_banks



    np.save(filename.replace('.wav', '.npy'),frames_features)

    return

def read_MFB(filename):
    #audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    #audio = audio.flatten()
    audio = np.load(filename.replace('.wav', '.npy'))
    return audio

class totensor(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            #img = torch.from_numpy(pic.transpose((0, 2, 1)))
            #return img.float()
            img = torch.FloatTensor(pic.transpose((0, 2, 1)))
            #img = np.float32(pic.transpose((0, 2, 1)))
            return img

            #img = torch.from_numpy(pic)
            # backward compatibility


class truncatedinputfromMFB(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, input_per_file=1):

        super(truncatedinputfromMFB, self).__init__()
        self.input_per_file = input_per_file

    def __call__(self, frames_features):

        network_inputs = []
        num_frames = len(frames_features)
        import random

        for i in range(self.input_per_file):

            j = random.randrange(NUM_PREVIOUS_FRAME, num_frames - NUM_NEXT_FRAME)
            if not j:
                frames_slice = np.zeros(NUM_FRAMES, FILTER_BANK, 'float64')
                frames_slice[0:(frames_features.shape)[0]] = frames_features.shape
            else:
                frames_slice = frames_features[j - NUM_PREVIOUS_FRAME:j + NUM_NEXT_FRAME]
            network_inputs.append(frames_slice)

        return np.array(network_inputs)

# %%

def read_audio(filename, sample_rate=SAMPLE_RATE):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.flatten()
    return audio
class truncatedinput(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __call__(self, input):

        #min_existing_frames = min(self.libri_batch['raw_audio'].apply(lambda x: len(x)).values)
        want_size = int(TRUNCATE_SOUND_FIRST_SECONDS * SAMPLE_RATE)
        if want_size > len(input):
            output = np.zeros((want_size,))
            output[0:len(input)] = input
            #print("biho check")
            return output
        else:
            return input[0:want_size]

def pre_process_inputs(signal=np.random.uniform(size=32000), target_sample_rate=8000,use_delta = USE_DELTA):
    filter_banks, energies = fbank(signal, samplerate=target_sample_rate, nfilt=FILTER_BANK, winlen=0.025)
    delta_1 = delta(filter_banks, N=1)
    delta_2 = delta(delta_1, N=1)

    filter_banks = normalize_frames(filter_banks)
    delta_1 = normalize_frames(delta_1)
    delta_2 = normalize_frames(delta_2)

    if use_delta:
        frames_features = np.hstack([filter_banks, delta_1, delta_2])
    else:
        frames_features = filter_banks
    num_frames = len(frames_features)
    network_inputs = []
    """Too complicated
    for j in range(c.NUM_PREVIOUS_FRAME, num_frames - c.NUM_NEXT_FRAME):
        frames_slice = frames_features[j - c.NUM_PREVIOUS_FRAME:j + c.NUM_NEXT_FRAME]
        #network_inputs.append(np.reshape(frames_slice, (32, 20, 3)))
        network_inputs.append(frames_slice)
        
    """
    import random
    j = random.randrange(NUM_PREVIOUS_FRAME, num_frames - NUM_NEXT_FRAME)
    frames_slice = frames_features[j - NUM_PREVIOUS_FRAME:j + NUM_NEXT_FRAME]
    network_inputs.append(frames_slice)
    return np.array(network_inputs)

class toMFB(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __call__(self, input):

        output = pre_process_inputs(input, target_sample_rate=SAMPLE_RATE)
        return output

# %% [markdown]
# # Agent 

# %% [markdown]
# ## Optimizer init

# %%
import torch.optim as optim

def create_optimizer(model, new_lr):
    # setup optimizer
    if config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=config.wd)
    elif config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=config.wd)
    elif config.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=config.lr_decay,
                                  weight_decay=config.wd)
    return optimizer


# %% [markdown]
# ## Evaluation Metrics

# %%
import numpy as np
from scipy import interpolate

def calculate_eer():
    
    return eer
def calculate_eer(fpr, tpr):
    fnr = 1 - tpr
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer


def evaluate(distances, labels):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 30, 0.01)
    tpr, fpr, accuracy = calculate_roc(thresholds, distances,
        labels)
    thresholds = np.arange(0, 30, 0.001)
    val,  far = calculate_val(thresholds, distances,
        labels, 1e-3)

    eer = calculate_eer(fpr, tpr)
    return tpr, fpr, accuracy, val,  far, eer


def calculate_roc(thresholds, distances, labels):

    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)

    tprs = np.zeros((nrof_thresholds))
    fprs = np.zeros((nrof_thresholds))
    acc_train = np.zeros((nrof_thresholds))
    accuracy = 0.0

    indices = np.arange(nrof_pairs)


    # Find the best threshold for the fold

    for threshold_idx, threshold in enumerate(thresholds):
        tprs[threshold_idx], fprs[threshold_idx], acc_train[threshold_idx] = calculate_accuracy(threshold, distances, labels)
    best_threshold_index = np.argmax(acc_train)



    return tprs[best_threshold_index], fprs[best_threshold_index], acc_train[best_threshold_index]


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, distances, labels, far_target=0.1):
    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)

    indices = np.arange(nrof_pairs)


    # Find the threshold that gives FAR = far_target
    far_train = np.zeros(nrof_thresholds)

    for threshold_idx, threshold in enumerate(thresholds):
        _, far_train[threshold_idx] = calculate_val_far(threshold, distances, labels)
    if np.max(far_train)>=far_target:
        f = interpolate.interp1d(far_train, thresholds, kind='slinear')
        threshold = f(far_target)
    else:
        threshold = 0.0

    val, far = calculate_val_far(threshold, distances, labels)


    return val, far


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    if n_diff == 0:
        n_diff = 1
    if n_same == 0:
        return 0,0
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

# %% [markdown]
# ## Train function

# %%
from tqdm import tqdm 

from torch.autograd import Variable

def train(train_loader, model, optimizer, epoch):
    # switch to train mode
    model.train()
    labels, distances = [], []

    # pbar = tqdm(enumerate(train_loader))
    pbar = enumerate(train_loader)
    for batch_idx, (data_a, data_p, data_n, label_p, label_n) in tqdm(enumerate(train_loader)):
        #print("on training{}".format(epoch))
        data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()
        data_a, data_p, data_n = Variable(data_a), Variable(data_p), \
                                 Variable(data_n)

        # compute output
        out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)


        if epoch > config.min_softmax_epoch:
            triplet_loss = TripletMarginLoss(config.margin).forward(out_a, out_p, out_n)
            loss = triplet_loss
            # compute gradient and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # logger.log_value('selected_triplet_loss', triplet_loss.data[0]).step()
            # #logger.log_value('selected_cross_entropy_loss', cross_entropy_loss.data[0]).step()
            # logger.log_value('selected_total_loss', loss.data[0]).step()

            # if batch_idx % config.log_interval == 0:
            #     pbar.set_description(
            #         'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tLoss: {:.6f}'.format(
            #             epoch, batch_idx * len(data_a), len(train_loader.dataset),
            #             100. * batch_idx / len(train_loader),
            #             loss.data[0]))

            if batch_idx % config.log_interval == 0:
                print('Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data_a), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader),
                            loss.data.item()))


            dists = l2_dist.forward(out_a,out_n) #torch.sqrt(torch.sum((out_a - out_n) ** 2, 1))  # euclidean distance
            distances.append(dists.data.cpu().numpy())
            labels.append(np.zeros(dists.size(0)))


            dists = l2_dist.forward(out_a,out_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
            distances.append(dists.data.cpu().numpy())
            labels.append(np.ones(dists.size(0)))



        else:
        # Choose the hard negatives
            d_p = l2_dist.forward(out_a, out_p)
            d_n = l2_dist.forward(out_a, out_n)
            all = (d_n - d_p < config.margin).cpu().data.numpy().flatten()

            # log loss value for mini batch.
            total_coorect = np.where(all == 0)
            # logger.log_value('Minibatch Train Accuracy', len(total_coorect[0]))

            total_dist = (d_n - d_p).cpu().data.numpy().flatten()
            # logger.log_value('Minibatch Train distance', np.mean(total_dist))

            hard_triplets = np.where(all == 1)
            if len(hard_triplets[0]) == 0:
                continue
            out_selected_a = Variable(torch.from_numpy(out_a.cpu().data.numpy()[hard_triplets]).cuda())
            out_selected_p = Variable(torch.from_numpy(out_p.cpu().data.numpy()[hard_triplets]).cuda())
            out_selected_n = Variable(torch.from_numpy(out_n.cpu().data.numpy()[hard_triplets]).cuda())

            selected_data_a = Variable(torch.from_numpy(data_a.cpu().data.numpy()[hard_triplets]).cuda())
            selected_data_p = Variable(torch.from_numpy(data_p.cpu().data.numpy()[hard_triplets]).cuda())
            selected_data_n = Variable(torch.from_numpy(data_n.cpu().data.numpy()[hard_triplets]).cuda())

            selected_label_p = torch.from_numpy(label_p.cpu().numpy()[hard_triplets])
            selected_label_n= torch.from_numpy(label_n.cpu().numpy()[hard_triplets])
            triplet_loss = TripletMarginLoss(config.margin).forward(out_selected_a, out_selected_p, out_selected_n)

            cls_a = model.forward_classifier(selected_data_a)
            cls_p = model.forward_classifier(selected_data_p)
            cls_n = model.forward_classifier(selected_data_n)

            criterion = nn.CrossEntropyLoss()
            predicted_labels = torch.cat([cls_a,cls_p,cls_n])
            true_labels = torch.cat([Variable(selected_label_p.cuda()),Variable(selected_label_p.cuda()),Variable(selected_label_n.cuda())])

            cross_entropy_loss = criterion(predicted_labels.cuda(),true_labels.cuda())

            loss = cross_entropy_loss + triplet_loss * config.loss_ratio
            # compute gradient and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # log loss value for hard selected sample
            # logger.log_value('selected_triplet_loss', triplet_loss.data[0]).step()
            # logger.log_value('selected_cross_entropy_loss', cross_entropy_loss.data[0]).step()
            # logger.log_value('selected_total_loss', loss.data[0]).step()
            # if batch_idx % config.log_interval == 0:
            #     pbar.set_description(
            #         'Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tLoss: {:.6f} \t # of Selected Triplets: {:4d}'.format(
            #             epoch, batch_idx * len(data_a), len(train_loader.dataset),
            #             100. * batch_idx / len(train_loader),
            #             loss.data[0],len(hard_triplets[0])))
            if batch_idx % config.log_interval == 0:
                print('Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tLoss: {:.6f} \t # of Selected Triplets: {:4d}'.format(
                    epoch, batch_idx * len(data_a), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.data.item(),len(hard_triplets[0])))
            
            dists = l2_dist.forward(out_selected_a,out_selected_n) #torch.sqrt(torch.sum((out_a - out_n) ** 2, 1))  # euclidean distance
            distances.append(dists.data.cpu().numpy())
            labels.append(np.zeros(dists.size(0)))


            dists = l2_dist.forward(out_selected_a,out_selected_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
            distances.append(dists.data.cpu().numpy())
            labels.append(np.ones(dists.size(0)))


    #accuracy for hard selected sample, not all sample.
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    tpr, fpr, accuracy, val, far, eer = evaluate(distances,labels)
    print('\33[91mTrain set: Accuracy: {:.8f}\n\33[0m'.format(np.mean(accuracy)))
    # logger.log_value('Train Accuracy', np.mean(accuracy))

    # do checkpointing
    # torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
    #             'optimizer': optimizer.state_dict()},
    #            '{}/checkpoint_{}.pth'.format("save_models", epoch))


# %% [markdown]
# ## Test function

# %%
def test(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    # pbar = tqdm(enumerate(test_loader))
    with torch.no_grad():
        pbar = enumerate(test_loader)
        for batch_idx, (data_a, data_p, label) in pbar:
            current_sample = data_a.size(0)
            data_a = data_a.resize_(config.test_input_per_file *current_sample, 1, data_a.size(2), data_a.size(3))
            data_p = data_p.resize_(config.test_input_per_file *current_sample, 1, data_a.size(2), data_a.size(3))
            if config.cuda:
                data_a, data_p = data_a.cuda(), data_p.cuda()
            # data_a, data_p, label = Variable(data_a, volatile=True), \
            #                         Variable(data_p, volatile=True), Variable(label)

            # compute output
            out_a, out_p = model(data_a), model(data_p)
            dists = l2_dist.forward(out_a,out_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
            dists = dists.data.cpu().numpy()
            dists = dists.reshape(current_sample,config.test_input_per_file).mean(axis=1)
            distances.append(dists)
            labels.append(label.data.cpu().numpy())

            # if batch_idx % config.log_interval == 0:
            #     # pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
            #     #     epoch, batch_idx * len(data_a), len(test_loader.dataset),
            #     #     100. * batch_idx / len(test_loader)))
            #     print('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
            #         epoch, batch_idx * len(data_a), len(test_loader.dataset),
            #         100. * batch_idx / len(test_loader)))

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])

        #print("distance {.8f}".format(distances))
        #print("distance {.1f}".format(labels))
        tpr, fpr, accuracy, val, far = evaluate(distances,labels)
        print('\33[91mTest set: Accuracy: {:.8f}\n\33[0m'.format(np.mean(accuracy)))
        print(f"Test epoch {epoch}: tpr={tpr} fpr={fpr} val={val} far={far}")
        # print(f"Test epoch {epoch}: Accuracy {np.mean(accuracy)}")
        # logger.log_value('Test Accuracy', np.mean(accuracy))

# %% [markdown]
# # Training
# 

# %%
# convert wav file to npy 
for key in train_data.keys():
    for file in train_data[key]:
        mk_MFB("./data/" + file)

# %%
for file1, file2, _ in test_pairs:
    mk_MFB(file1)
    mk_MFB(file2)

# %%
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

file_loader = read_MFB
transform = transforms.Compose([
    truncatedinputfromMFB(),
    totensor()
])
transform_T = transforms.Compose([
    truncatedinputfromMFB(input_per_file=config.test_input_per_file),
    totensor()
])
train_dir = DeepSpeakerDataset(train_data, config.n_triplets, file_loader, transform)
test_dir = TestDataset(test_pairs, file_loader, transform_T)

# %%
transform_T = transforms.Compose([
    truncatedinputfromMFB(input_per_file=config.test_input_per_file),
    totensor()
])
test_dir = TestDataset(test_pairs, file_loader, transform_T)

# %%
model = DeepSpeakerModel(embedding_size = config.embedding_size,
                    num_classes=len(train_dir.classes))

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model.cuda()

# %%

l2_dist = PairwiseDistance(2)

# %%
optimizer = create_optimizer(model, config.lr)

# %%
kwargs = {'num_workers': 0, 'pin_memory': True} if config.cuda else {}
train_loader = DataLoader(train_dir, batch_size=config.batch_size, shuffle=False, **kwargs)

# %%
test_loader = DataLoader(test_dir, batch_size=config.test_batch_size, shuffle=False, **kwargs)

# %%
start = config.start_epoch
end = start + config.epochs

# %%
for epoch in range(start, end):
    train(train_loader, model, optimizer, epoch)
    test(test_loader, model, epoch)

# %%



