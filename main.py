from models import ModelHandler
from data import DataHandler
import wandb
import config_parser

def main():

    config = config_parser.load_config('config.json')
    # Paths and configuration parameters
    

    holdout_trial_site = config["holdout_trial_site"]
    exp_name = holdout_trial_site
    
    wandb.login()
    wandb.init(config=config, project=exp_name, entity="msikarou")

    dataloaders = DataHandler.get_dataloaders(holdout_trial_site, 
                                       config['augmentation_in_training'], 
                                       config['batch_size'], 
                                       num_workers=64)
    
    device = ModelHandler.get_device()
    model = ModelHandler.get_model(config, device)
    model = ModelHandler.freeze_layers(model, freeze_until_layer=2)
    optimizer, scheduler = ModelHandler.initialize_optimizer(model, config)
    criterion = ModelHandler.get_criterion('cross_entropy')
    model = ModelHandler.train_model(model, criterion, optimizer, scheduler, 
                             dataloaders, device, 
                             num_epochs=config['epochs'])

    save_path = '_'.join(['Experiment_' + holdout_trial_site, 
                          model.name, 
                          str(config['learning_rate']), 
                          str(config['epochs']), 
                          str(config['batch_size']), 
                          str(config['augmentation_in_training']), 
                          ".pt"])
    ModelHandler.save_model(model, save_path)

if __name__ == "__main__":
    main()
