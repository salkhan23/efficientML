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
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
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


def evaluate(model, data_loader, device):
    model.eval()

    n_samples = 0
    n_correct = 0

    for inputs, labels in tqdm(data_loader, desc="eval", leave=False,):
        # Move the data from CPU to GPU
        inputs = inputs.to(device)
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


def main(b_size, random_seed=10, data_dir='./data', save_dir='./results_trained_models', n_epochs=10, lr=1e-2):

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

    for epoch in range(n_epochs):

        running_loss = 0.0
        epoch_start_time = datetime.now()
        n_samples = 0
        n_correct = 0

        for b_idx, (data, labels) in \
                tqdm(enumerate(train_loader, 0), total=len(train_loader), desc=f'Epoch {epoch+1}/{n_epochs}'):

            data = data.to(device)
            labels = labels.to(device)

            model_out = net(data)

            # Compute the loss using the raw outputs and the true labels
            loss = criterion(model_out, labels)

            # Zero the gradients before backward pass
            optimizer.zero_grad()
            # Backward pass: compute the gradients of the loss with respect to model parameters
            loss.backward()
            # Update the model parameters based on the computed gradients
            optimizer.step()

            # Train Acc
            predictions = torch.argmax(model_out, dim=1)
            n_samples += labels.size(0)
            n_correct += (predictions == labels).sum()

            running_loss += loss.item()
            if b_idx % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {b_idx+ 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        lr_scheduler.step()

        test_acc = evaluate(net, test_loader, device)
        train_acc = n_correct / n_samples * 100
        print(
            f"Epoch {epoch} took {datetime.now()- epoch_start_time}. Train Acc {train_acc:0.2f}, "
            f"Test Acc {test_acc:0.2f}")

        # Save the model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model = net.state_dict()

    end_time = datetime.now()
    total_training_time = end_time - train_start_time
    print(f"Total Training Time {total_training_time}")

    if best_model:
        save_model(
            model=net,
            best_model_state_dict=best_model,
            save_dir=save_dir,
            n_epochs=n_epochs,
            best_test_acc=best_test_acc
        )

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
