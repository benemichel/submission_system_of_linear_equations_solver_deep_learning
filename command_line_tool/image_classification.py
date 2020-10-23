import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
import torch
from torchvision import datasets, transforms
from PIL import Image, ImageOps
import os
import cv2 as cv
from PIL import Image, ImageOps

class ConvNN(nn.Module):
    """
    custom model for math symbol classification
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        '''Defines layers of a neural network.
        params:
           input_dim: Input layer dimensions (features)
           hidden_dim: Hidden layer dimensions
           output_dim: Output dimensions
         '''
        super(ConvNN, self).__init__()

        #The parameters kernel_size, stride, padding, dilation can either be:
        #a single int – in which case the same value is used for the height and width dimension
        in_channels = [1, 4, 8]  #first must fit to input data channel dimensions (1)
        out_channels = [4, 8, 16]
        kernel_sizes = [3, 3, 3]
        conv_strides = [1, 1, 1]
        pool_strides = [3, 3, 3]
        paddings = [1, 1, 1] #pad should be smaller than half of kernel size
        pool_kernel_sizes = [3, 3, 3]
        #input dims for first conv layer (used to calc output dims of that layer)
        self.in_dims = input_dim
        self.activation = nn.ReLU()
        #for fully connected layers (last output dim from conv layer)
        self.conv_out_channel = out_channels[-1]

        conv_layers = []
        for i in range(len(kernel_sizes)):

            conv_layers.append(nn.Conv2d(
                                    in_channels = in_channels[i],
                                    out_channels = out_channels[i],
                                    kernel_size = kernel_sizes[i],
                                    stride = conv_strides[i],
                                    padding = paddings[i])
                              )

            conv_layers.append(self.activation)

            #calculate input dimensions for next layer
            out_dims = self.calc_out_dims(
                                self.in_dims,
                                conv_strides[i],
                                paddings[i],
                                kernel_sizes[i],
                                debug = False
                                )

            print(f"first conv layer output dims: {out_dims}")

            conv_layers.append(nn.MaxPool2d(
                                    kernel_size = pool_kernel_sizes[i],
                                    stride = pool_strides[i],
                                    padding = paddings[i])
                              )

            conv_layers.append(nn.Dropout(dropout_rate))

            out_dims = self.calc_out_dims(
                                out_dims,
                                pool_strides[i],
                                paddings[i],
                                pool_kernel_sizes[i],
                                debug = False
                                )

            print(f"max pool layer output dims: {out_dims}")

            self.in_dims = out_dims

        self.conv_layers = nn.Sequential(*conv_layers)

        #dimensions for fc layer
        print(f"in dims for fc layer : {self.conv_out_channel} * {self.in_dims} * {self.in_dims}")

        #fully connected layers
        fc_layers = []
        fc_layers.append(nn.Linear(self.conv_out_channel * self.in_dims * self.in_dims, hidden_dim))
        fc_layers.append(self.activation)
        fc_layers.append(nn.Dropout(dropout_rate))
        fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        fc_layers.append(self.activation)
        fc_layers.append(nn.Dropout(dropout_rate))
        fc_layers.append(nn.Linear(hidden_dim, output_dim))

        self.fc_layers = nn.Sequential(*fc_layers)


    def calc_out_dims(self, in_dim, stride, padding, kernel_size, dilation = 1, debug = False):
        """
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        H and W same dimensions f dilatione etc. are only one int
        dilation is 1 by default
        """

        out_dim = int(((in_dim + 2 * padding - dilation * (kernel_size - 1) -1) / stride) + 1)
        if debug:
            print("")
            print("")
            print(f" in + 2 * padding - kernel_size                                 ")
            print(f" -------------------------------    + 1   = dim_out             ")
            print(f"                stride                                          ")
            print("")
            print(f" {in_dim} + 2 * {padding} - {dilation} * {kernel_size}                      ")
            print(f" -----------------------------------------------------   + 1   =  {out_dim} ")
            print(f"                      {stride}                                              ")
            print("")
            print("")

        return out_dim


    def forward(self, out):
        '''Feedforward sequence
        params:
            out: input data which will be transformed to model output
         '''

        out = self.conv_layers(out)
        out = out.view(-1, self.conv_out_channel * self.in_dims * self.in_dims)
        out = self.fc_layers(out)

        return out

def train(model, trainloader, testloader, epochs, optimizer, criterion, device):
    """
    This is the training method that is called by the PyTorch training script.
    This method uses early stopping.
    params:
        model        - The PyTorch model that we wish to train.
        train_loader - The PyTorch DataLoader that should be used during training.
        epochs       - The total number of epochs to train for.
        optimizer    - The optimizer to use during training.
        criterion    - The loss function used for training.
        device       - Where the model and data should be loaded (gpu or cpu).
    """

    train_losses = []
    val_losses = []
    early_stopped = False
    min_val_loss_factor = 0.9  #val loss has to decrease by 10% to avoid early stopping
    early_stopping_epochs = 3   #no improve (condition see above) for n epochs
    epochs_wo_improvement = 0
    best_model = {}
    save_model_path = './model/model_dict.pth'
    epoch_stopped = 0

    print("### Start training ###")
    for epoch in range(1, epochs + 1):

        train_loss_total = 0
        val_loss_total = 0

        #set model to training mode
        model.train()
        for i, batch in enumerate(trainloader, 0):
            #move it (model and data tensors) to GPU if available (.to(cuda))
            batch_inputs, batch_labels = batch
            inputs, labels = batch_inputs.to(device, dtype=torch.float), batch_labels.to(device)

            #zero the gradient parameters
            optimizer.zero_grad()

            # get output of model (forawrd + backward + optimize)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()

        #validate
        model.eval()
        for i, batch in enumerate(testloader, 0):
            batch_inputs, batch_labels = batch
            inputs, labels = batch_inputs.to(device, dtype=torch.float), batch_labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss_total += loss.item()

        train_loss = train_loss_total / len(trainloader)
        val_loss = val_loss_total / len(testloader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch: {epoch}, Train Loss: {train_loss} Val Loss: {val_loss}")

        #early stopping
        if epoch > 1 and (val_loss > val_losses[epoch - 2] * min_val_loss_factor) and not early_stopped:
            print("no improvement")
            epochs_wo_improvement += 1

        if epochs_wo_improvement >= early_stopping_epochs and not early_stopped:
            print("### Early stopping ###")
            early_stopped = True
            epoch_stopped = epoch
            #save best model
            torch.save(model.state_dict(), save_model_path)

    if not early_stopped and epoch == epochs:
        #save last model
        torch.save(model.state_dict(), save_model_path)

    print("###  Training Finished  ###")

    return (save_model_path, epoch_stopped, train_losses, val_losses)

def predict(data, model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    # Turn on evaluation mode
    model.eval()
    with torch.no_grad():
        out = model(data.float())
        result = out.cpu().detach().numpy()

    return result

class CustomDataset(torch.utils.data.Dataset):
    """Load image files from one folder, filename as label"""

    def __init__(self, root_dir, loader = None, transform=None):
        """
        params:
            root_dir : Directory with all the images.
            loader: preprocessing
            transform: optional transformation (e.g. toTensor)
        """
        self.files = os.listdir(root_dir)
        self.root_dir = root_dir
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.root_dir,
                                self.files[idx])

        image = None

        if self.loader:
            image = self.loader(image_path)
        else:
            image = cv.imread(image_path)

        if self.transform:
            image = self.transform(image)

        sample = (image, self.files[idx])

        return sample
def create_pytorch_dataset_from_folder(directory):
    transform_scan = transforms.Compose([ transforms.ToTensor() ])

    dataset_scan = CustomDataset(root_dir = directory,
                                transform = transform_scan,
                                loader = preprocess_scan_image,
                                )

    print(f"dataset created with {len(dataset_scan)} datapoints")

    return dataset_scan

def preprocess_scan_image(path):
    """preprocess iamges before creating tensors"""

    with Image.open(path) as image:

        image_resized = image.resize((28,28))
        image_gray = cv.cvtColor(np.array(image_resized), cv.COLOR_BGR2GRAY)
        image_norm = np.array(image_gray)/np.max(np.array(image_gray))
        thresh = 0.1
        image_threshed = image_norm
        image_threshed[image_threshed < thresh] = 0
        image_threshed[image_threshed >= thresh] = 1

    return image_threshed

def predict_sle_symbols(model, dataloader, mappings, reverse_mappings, plot = False):
    """"""

    n = len(dataloader)
    dataiter = iter(dataloader)
    predictions = {}

    for i in range(n):
        image, filename = dataiter.next()
        filename = filename[0]

        prediction = predict(image, model)
        prediction_label = reverse_mappings[np.argmax(prediction)]
        predictions[filename.replace('.jpg', '')] = prediction_label

        if plot:
            print(f"prediction for image {filename}:")
            plt.bar(x = mappings.keys(), height = prediction.squeeze())
            plt.show()
            plt.imshow(image.squeeze().numpy())
            plt.show()

    #format
    print(predictions)

    return predictions


def create_sle_parts(line_prediction):
    #extract lists for chars in each line
    n_lines = int(max([index.split('_')[0] for index in list(line_prediction.keys())])) + 1

    line_indices = []
    for i in range(n_lines):
        indices = [key for key in line_prediction.keys() if key.split('_')[0] == str(i)]
        line_indices.append(indices)

    line_indices

    line_symbols = []
    for line_i, line in enumerate(line_indices):
        symbols = []
        n = len(line)
        for column_i in range(n):
            symbols.append(line_prediction[str(line_i) + '_' + str(column_i)])

        line_symbols.append(symbols)

    for line in line_symbols:
        print("".join(line))


    #create A and b
    #get index of '=' (last of multiple)
    b = np.full(n_lines, fill_value = 'xxx', dtype=object)
    A = np.full((n_lines, n_lines), fill_value = 'xxx', dtype=object)

    line_var_indices = []
    var_list = ['x', 'y', 'z']
    for line_i, line in enumerate(line_symbols):
        try:
            index_equal = ''.join(line).rindex('=')
            b[line_i] = ''.join(line[index_equal + 1:])
            #print(''.join(line[index_equal + 1:]))

            var_indices = []
            for var in var_list :
                if var in line[:index_equal]:
                    var_indices.append(( line[:index_equal].index(var), var))

            var_indices_sorted = sorted(var_indices)
            line_var_indices.append(var_indices_sorted)

        except:
            print("= sign could not be found")


    #print(line_var_indices)

    for line_i, line in enumerate(line_var_indices):
        last_index = 0
        for index, var in line:
            param = ''.join(line_symbols[line_i][last_index:index])
            #there must be max one character of '-' and '+', if multiple
            #creation of mathematical param will fail
            try:
                A[line_i, var_list.index(var)] = param
            except:
                print("problem with creation of A")
            last_index = index + 1


    return line_symbols, A, b, var_list

def calc_solution(A, b):

    try:
        #if one element is 'xxx' this will fail
        A = A.astype('int32')
        b = b.astype('int32')
        print(A)
        print(b)
        A_inv = np.linalg.inv(A)
        #round solution
        solution = np.matrix.round(np.matmul(A_inv , np.array(b)), decimals =  0)
        print("###solution found ####")
    except:
        print("solution could not be calculated")
        solution = np.array(['xxx', 'xxx', 'xxx'], dtype = 'object')

    return solution
