
from src.training.config import TrainingConfig


def adjust_lr(optimizer, scheduler, epoch, config: TrainingConfig):
    """
    Adjust the optimizer's learning rate based on a predefined schedule.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be adjusted.
        scheduler (torch.optim.lr_scheduler._LRScheduler): A PyTorch learning rate scheduler 
            used in some schedules (e.g. 'TST').
        epoch (int): The current epoch number.
        config (TrainingConfig): Configuration object containing the learning rate adjustment
    """
    # Initialize a dictionary for the new learning rate, keyed by the current epoch if applicable.
    lr_adjust = {}

    # Use the base learning rate from args as a reference.
    base_lr = config.learning_rate

    # Define schedules based on the value in config.learning_rate_adjustment
    if config.learning_rate_adjustment == 'type1':
        # Halve the learning rate each epoch
        lr_adjust = {epoch: base_lr * (0.5 ** ((epoch - 1) // 1))}
    elif config.learning_rate_adjustment == 'type2':
        # Manually specify learning rate at particular epochs
        lr_adjust = {
            2: 5e-5,  4: 1e-5,  6: 5e-6,  8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif config.learning_rate_adjustment == 'type3':
        # Keep the base LR for the first 3 epochs, then multiply by (0.9^(epoch - 3))
        lr_adjust = {
            epoch: base_lr if epoch < 3 else base_lr * (0.9 ** ((epoch - 3) // 1))
        }
    elif config.learning_rate_adjustment == 'constant':
        # Keep the learning rate constant throughout
        lr_adjust = {epoch: base_lr}
    elif config.learning_rate_adjustment == '3':
        # Keep base LR until epoch 10, then reduce by 10x
        lr_adjust = {epoch: base_lr if epoch < 10 else base_lr * 0.1}
    elif config.learning_rate_adjustment == '4':
        # Keep base LR until epoch 15, then reduce by 10x
        lr_adjust = {epoch: base_lr if epoch < 15 else base_lr * 0.1}
    elif config.learning_rate_adjustment == '5':
        # Keep base LR until epoch 25, then reduce by 10x
        lr_adjust = {epoch: base_lr if epoch < 25 else base_lr * 0.1}
    elif config.learning_rate_adjustment == '6':
        # Keep base LR until epoch 5, then reduce by 10x
        lr_adjust = {epoch: base_lr if epoch < 5 else base_lr * 0.1}
    elif config.learning_rate_adjustment == 'TST':
        # Use the learning rate from the scheduler's last update
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    # If the epoch is specified in the adjustment schedule, update the LR
    if epoch in lr_adjust:
        new_lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
            print(f"Updating learning rate to {new_lr}")
