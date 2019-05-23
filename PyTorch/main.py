from model.model import ClassificationModel
from utils.utils import get_config_from_json
from data_loader.data_loader import ClassificationTrainDataset, ClassificationValDataset
from torch.utils.data import DataLoader
from trainer.trainer import trainClassification

def mainClassification():

    # Get configurations
    config = get_config_from_json('config/classification.config')


    # Loading Model
    model = ClassificationModel(config)
    model.load()

    count = 0
    for param in model.parameters():
        if count < 27:
            print(param)
            param.requires_grad = False
            count += 1

    print(model)

    # Loading Dataset
    train_dataset = ClassificationTrainDataset(config)
    val_dataset = ClassificationValDataset(config)

    # Creating Data Loader
    train_loader = DataLoader(dataset = train_dataset,
                              batch_size = config['model']['batch_size'])

    val_loader = DataLoader(dataset = val_dataset,
                            batch_size = config['model']['batch_size'])

    # Train model
    trainClassification(config, model, train_loader, val_loader)



if __name__ == "__main__":
    mainClassification()
