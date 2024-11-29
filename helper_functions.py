import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
def evaluate_classifier(classifier, dataloader):
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = classifier(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


def _get_next_run_index(log_dir):
    """Determine the next run index based on the existing runs in the directory."""
    if not os.path.exists(log_dir):
        return 0
    existing_runs = [int(f.split('_')[-1]) for f in os.listdir(log_dir) if f.startswith('run_') and f.split('_')[-1].isdigit()]
    return max(existing_runs, default=-1) + 1

def train_convnet(
    model: torch.nn.Module, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    learning_rate_scheduler, 
    num_epochs=5, 
    filename='model.pth'
):
    # Determine the logging directory and next run index
    base_log_dir = f"tensorboard_logs/{os.path.splitext(os.path.basename(filename))[0]}"
    k = _get_next_run_index(base_log_dir)
    writer = SummaryWriter(log_dir=f"{base_log_dir}/run_{k}")
     
    # Check if models directory exists, if not, create it
    if not os.path.exists("models"):
        os.makedirs("models")
    
    best_accuracy = 0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=True)
        for images, labels in train_bar:
            images, labels = images.cuda(), labels.cuda()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_accuracy = 100 * correct / total
        average_train_loss = total_loss / len(train_loader)

        # Log training metrics for the epoch
        writer.add_scalar("Loss/Train", average_train_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=True)
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.cuda(), labels.cuda()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Update metrics
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate validation accuracy/loss for the epoch
        val_accuracy = 100 * val_correct / val_total
        average_val_loss = val_loss / len(val_loader)

        # Log validation metrics for the epoch
        writer.add_scalar("Loss/Validation", average_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)

        # Save the model if validation accuracy improves
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(),os.path.join('models', filename))

        # Adjust learning rate
        learning_rate_scheduler.step()

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {average_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    writer.close()
