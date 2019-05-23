from torch.autograd import Variable
import torch.nn as nn
import torch



def trainClassification(config, model, train_loader, val_loader):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=config['model']['learning_rate'])

    print(config)

    for epoch in range(config['model']['num_epochs']):

        total, correct = 0, 0
        for images, labels in train_loader:
            images = Variable(images)
            labels = Variable(labels)
            optimizer.zero_grad()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (labels == predicted).sum(0)
            train_loss = criterion(outputs, labels)

            train_loss.backward()
            optimizer.step()
            
        train_accuracy = 100.0 * correct / total
        print('train_accuracy : ', train_accuracy)

        total, correct = 0, 0
        for images, labels in val_loader:
            images = Variable(images)
            labels = Variable(labels)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (labels == predicted).sum(0)
            val_loss = criterion(outputs, labels)

        val_accuracy = 100.0 * correct / total

        print('Epoch: %2d/%2d \t Train Loss: %.4f \t Train Acc: %.3f \t Val Loss: %.3f \t Val Acc: %.3f' % (epoch+1, config['model']['num_epochs'], train_loss.item(), train_accuracy, val_loss.item(), val_accuracy))

        torch.save( model.state_dict(), 'checkpoint/classificationModel.pth')
