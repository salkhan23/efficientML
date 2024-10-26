# ---------------------------------------------------------------------------------------
# Train Cifar-10 Model from scratch
# TODO: current highest test accuracy is 81.06. Can go up to 92.95. Explore Improvements
# ---------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from vgg import VGG
from tqdm import tqdm


def show_image(img, title=None):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)


def get_cifar10_datasets_and_data_loaders(data_dir, b_size):
    """
    Loads the CIFAR-10 dataset and creates data loaders for training and testing.

    :param data_dir: dir where the CIFAR-10 dataset is/will be stored.
    :param b_size: Batch Size

    :return: tuple containing
        training dataset,
        training data loader,
        testing dataset,
        testing data loader,
        and class names.
    """
    transforms_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # normalizes inputs to range -1, 1
    ])

    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # normalizes inputs to range -1, 1
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms_train
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=b_size,
        shuffle=True,
        num_workers=12
    )

    test_set = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms_test
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=b_size,
        shuffle=False,
        num_workers=12)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_set, train_loader, test_set, test_loader, classes


def evaluate(model, data_loader, device, pre_process_cbs=None):
    """
    Evaluates a model's performance on a given dataset.

    This function runs the model in evaluation mode on a dataset provided by the data loader.
    It optionally applies preprocessing callbacks on the input data, performs inference, and
    computes the accuracy of the model by comparing predicted outputs to the ground truth labels.

    :param model:
    :param data_loader:
    :param device:
    :param pre_process_cbs: A list of preprocessing callback functions  applied sequentially to each batch of inputs.
           Defaults to None.

    :return:
    """
    model.eval()

    n_samples = 0
    n_correct = 0

    for inputs, labels in tqdm(data_loader, desc="eval", leave=False,):
        # Move the data from CPU to GPU
        inputs = inputs.to(device)

        inputs = inputs.cuda()
        if pre_process_cbs is not None:
            for preprocess in pre_process_cbs:
                inputs = preprocess(inputs)

        targets = labels.to(device)

        # Inference
        outputs = model(inputs)

        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)

        # Update metrics
        n_samples += labels.size(0)
        n_correct += (outputs == targets).sum()

    return (n_correct / n_samples * 100).item()


def save_model(model, best_model_state_dict, save_dir, best_test_acc, n_epochs):

    save_file_name = \
        f'./{datetime.now().strftime("%d-%m-%Y")}_{model.__class__.__name__}' \
        f'_net_train_epochs_{n_epochs}_acc_{int(best_test_acc)}.pth'

    os.makedirs(save_dir, exist_ok=True)
    save_file_path = os.path.join(save_dir, save_file_name)
    torch.save(best_model_state_dict, save_file_path)

    print(f'Model saved to {save_file_path}')


def train(
        model, dataloader, criterion, optimizer, scheduler, callbacks=None):
    """
    Train model for one epoch.

    :param model: nn.Module - The neural network model to train.
    :param dataloader: DataLoader - The dataloader providing training data.
    :param criterion: nn.Module - The loss function used for optimization.
    :param optimizer: optim.Optimizer - The optimizer for training the model.
    :param scheduler: LambdaLR - The learning rate scheduler for updating the learning rate.
    :param callbacks: list, optional - A list of callback functions to be called during training (default: None).
    :return: None
    """
    model.train()

    n_samples = 0
    n_correct = 0
    train_acc = 0

    for inputs, targets in tqdm(dataloader, desc='train', leave=False):
        # Move the data from CPU to GPU
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Reset the gradients (from the last iteration)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward propagation
        loss.backward()

        # Update optimizer
        optimizer.step()

        # Train accuracy
        predictions = torch.argmax(outputs, dim=1)
        n_samples += targets.size(0)
        n_correct += (predictions == targets).sum()

        # Execute any provided callbacks
        if callbacks is not None:
            for callback in callbacks:
                callback()

    train_acc = n_correct / n_samples * 100

    # update the learning rate after each epoch
    scheduler.step()

    return train_acc


def main(b_size, random_seed=10, data_dir='./data', save_dir=None, n_epochs=10, lr=1e-2):

    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_set, train_loader, test_data_set, test_loader, _ = (
        get_cifar10_datasets_and_data_loaders(data_dir=data_dir, b_size=b_size))

    # Debug ---------------------------------------------------------------------------------------
    # # Display a single Image
    # disp_img, disp_img_class = train_loader.dataset[0]
    # show_image(disp_img, classes[disp_img_class])
    #
    # # Display a bunch of images
    # dataiter = iter(train_loader)
    # images, labels = next(dataiter)  # returns batches
    # f = plt.figure()
    # show_image(torchvision.utils.make_grid(images))
    # f.suptitle(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    # Model ---------------------------------------------------------------------------------------
    net = VGG()
    net.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Training ------------------------------------------------------------------------------------
    train_start_time = datetime.now()

    best_test_acc = 0
    best_model = None

    # Track  the best model and accuracy
    best_model_state = None
    best_test_acc = 0.0

    # Start timer for training duration
    training_start_time = datetime.now()

    print(f"Starting Training for {n_epochs} epochs")

    test_acc = 0
    train_acc = 0

    # Training Loop for each epoch
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}. Starting accuracies: Train={train_acc:0.2f}, Test={test_acc:0.2f}")

        # Train the model for one epoch using the train function
        train_acc = train(
            model=net,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            callbacks=None
        )

        # Evaluate the model on the test set after each epoch
        test_acc = evaluate(net, test_loader, device)
        # print(f"Test Accuracy after Epoch {epoch + 1}: {test_acc:.2f}%")

        # Check if this is the best model so far
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = net.state_dict()
            # print(f"New Best Model found at Epoch {epoch + 1} with Test Accuracy: {best_test_acc:.2f}%")

    # Save the best model to the specified directory
    if save_dir is not None and best_model_state is not None:
        net.load_state_dict(best_model_state)  # Load best model state
        save_model(model=net, save_dir=save_dir, n_epochs=n_epochs, best_test_acc=best_test_acc)
        print(f"Best Model Saved with Test Accuracy: {best_test_acc:.2f}%")

    # Calculate and print the total training duration
    training_duration = datetime.now() - training_start_time
    print(f"Total Training Duration: {training_duration}")

    print("Training Completed!")

    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    plt.ion()
    batch_size = 128
    data_directory = './data/cifar10'

    if torch.cuda.is_available():
        print("CUDA is available. Moving model to GPU...")
    else:
        print("CUDA is not available. Moving model to CPU...")

    main(
        b_size=batch_size,
        data_dir=data_directory,
        n_epochs=100,
        lr=1e-2
    )
