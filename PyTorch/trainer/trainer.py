from torch.autograd import Variable
import torch.nn as nn
import torch



def trainClassification(config, model, train_loader, val_loader):

    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    print(config)

    train_acc_arr, val_acc_arr, train_loss_arr, val_loss_arr = [], [], [], []

    for epoch in range(config['model']['num_epochs']):
        
        model.train() 
        loss_train, loss_val = 0, 0
        total, correct = 0, 0

        for i, (images, labels) in enumerate(train_loader):
            
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images)
                labels = Variable(labels)


            optimizer.zero_grad()

            outputs = model(images)
            train_loss = criterion(outputs, labels)


            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if torch.cuda.is_available():
                correct += (labels.cpu() == predicted.cpu()).sum(0)
            else:
                correct += (labels == predicted).sum(0)
            loss_train += train_loss.item()*images.size(0)


            train_loss.backward()
            optimizer.step()
            
            if i%100 == 0:
                print('[%d/%d] \t Train Loss: %.3f \t Train Acc: %.3f   %d/%d'   % (i,
                                                                                    len(train_loader),
                                                                                    loss_train/(i+1),
                                                                                    100.0*float(correct)/total,
                                                                                    correct,
                                                                                    total ))
    
                
        train_accuracy = 100.0 * float(correct) / total

        model.eval()
        total, correct = 0, 0

        for images, labels in val_loader:
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images)
                labels = Variable(labels)

            outputs = model(images)
            val_loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            if torch.cuda.is_available():
                correct += (labels.cpu() == predicted.cpu()).sum(0)
            else:
                correct += (labels == predicted).sum(0)


            loss_val += val_loss.item()*images.size(0)

        val_accuracy = 100.0 * correct / total

        print('Epoch: %2d/%2d \t Train Loss: %.8f \t Train Acc: %.3f \t Val Loss: %.8f \t Val Acc: %.3f' \
               % (epoch+1, 
                  config['model']['num_epochs'], 
                  loss_train/len(train_loader), 
                  train_accuracy, 
                  loss_val/len(val_loader), 
                  val_accuracy))

        train_acc_arr.append(train_accuracy)
        train_loss_arr.append( loss_train/len(train_loader) )
        val_acc_arr.append(val_accuracy)
        val_loss_arr.append( loss_val/len(val_loader) )

        torch.save( model.state_dict(), config['model']['checkpoint'])

    print('Train Loss : ', train_loss_arr)
    print('Val Loss : ', val_loss_arr)
    print('Train Acc : ', train_acc_arr)
    print('Val Acc : ', val_acc_arr) 







def trainLRCN(config, model, train_loader, val_loader):


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['learning_rate'])

    print(config)

    for epoch in range(6, config['model']['num_epochs']):

        total, correct = 0, 0
        loss_train, loss_val = 0, 0

        model.train() 
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images)
                labels = Variable(labels)


            optimizer.zero_grad()

            outputs = model(images)
            train_loss = criterion(outputs, labels.long()) 


            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if torch.cuda.is_available():
                correct += (labels.cpu() == predicted.cpu()).sum(0)
            else:
                correct += (labels == predicted).sum(0)            
            loss_train += train_loss.item() * images.size(0)


            train_loss.backward()
            optimizer.step()


            if i%5 == 0:
                print('[%d/%d] \t Train Loss: %.3f \t Train Acc: %.3f' % (i,
                                                                          len(train_loader),
                                                                          loss_train/(i+1),
                                                                          100.0*float(correct)/total ))

            if i%50 == 0:
                torch.save( model.state_dict(), config['model']['checkpoint'])
                print('Model saved to : %s' % config['model']['checkpoint'])


        train_accuracy = 100.0 * float(correct) / total

        model.eval() 
        total, correct = 0, 0
        for images, labels in val_loader:
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images = Variable(images)
                labels = Variable(labels)

            outputs = model(images)
            val_loss = criterion(outputs, labels.long())


            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            if torch.cuda.is_available():
                correct += (labels.cpu() == predicted.cpu()).sum(0)
            else:
                correct += (labels == predicted).sum(0)

                
            loss_val += val_loss.item() * images.size(0)

        val_accuracy = 100.0 * correct / total
        print('Epoch: %2d/%2d \t Train Loss: %.4f \t Train Acc: %.3f \t Val Loss: %.3f \t Val Acc: %.3f' \
               % (epoch+1, 
                  config['model']['num_epochs'], 
                  loss_train/len(train_loader), 
                  train_accuracy, 
                  loss_val/len(val_loader), 
                  val_accuracy))

        torch.save( model.state_dict(), config['model']['checkpoint'])

