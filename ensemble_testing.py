import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from dataloader import get_cifar100_dataloader
from training import select_model
from tqdm import tqdm
import os


class EnsembleModel:
    def __init__(self, weight_config='best', device='cuda'):
        self.device = device
        self.weight_config = weight_config
        self.models = {
            'alexnet': select_model('alexnet'),
            'vgg16': select_model('vgg16'),
            'resnet18': select_model('resnet18')
        }

        # Load the trained models based on weight configuration
        suffix = '_best.pth' if weight_config == 'best' else '_epoch5.pth'
        weights_dir = 'weights'  # New weights directory

        for name, model in self.models.items():
            weight_path = os.path.join(weights_dir, f'{name}{suffix}')
            try:
                model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
                model.to(device)
                model.eval()
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not find trained model weights for {weight_path}")

    def _get_softmax_outputs(self, image):
        """
        Get softmax outputs from all models for a single image
        """
        outputs = {}
        with torch.no_grad():
            for name, model in self.models.items():
                logits = model(image)
                outputs[name] = F.softmax(logits, dim=1)
        return outputs

    def max_probability(self, image):
        """
        Strategy A: Maximum probability across all models
        """
        outputs = self._get_softmax_outputs(image)

        # Stack all probabilities and find maximum
        all_probs = torch.stack([out for out in outputs.values()])
        max_probs, _ = torch.max(all_probs, dim=0)
        predicted_label = torch.argmax(max_probs, dim=1)

        return predicted_label

    def probability_averaging(self, image):
        """
        Strategy B: Average probabilities across all models
        """
        outputs = self._get_softmax_outputs(image)

        # Stack and average probabilities
        all_probs = torch.stack([out for out in outputs.values()])
        avg_probs = torch.mean(all_probs, dim=0)
        predicted_label = torch.argmax(avg_probs, dim=1)

        return predicted_label

    def majority_voting(self, image):
        """
        Strategy C: Majority voting based on top predictions
        """
        outputs = self._get_softmax_outputs(image)

        # Get top prediction from each model
        predictions = []
        for output in outputs.values():
            pred = torch.argmax(output, dim=1)
            predictions.append(pred)

        # Stack predictions and find mode (majority vote)
        stacked_preds = torch.stack(predictions)
        predicted_label = torch.mode(stacked_preds, dim=0).values

        return predicted_label


def evaluate_ensemble(ensemble_model, test_loader, weight_config, device='cuda'):
    """
    Evaluate the ensemble model using all three strategies
    """
    strategies = {
        'Max Probability': ensemble_model.max_probability,
        'Probability Averaging': ensemble_model.probability_averaging,
        'Majority Voting': ensemble_model.majority_voting
    }

    results = {name: {'correct': 0, 'total': 0} for name in strategies.keys()}

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f'Evaluating ensemble ({weight_config} weights)'):
            images, labels = images.to(device), labels.to(device)

            # Evaluate each strategy
            for name, strategy in strategies.items():
                predictions = strategy(images)
                results[name]['correct'] += (predictions == labels).sum().item()
                results[name]['total'] += labels.size(0)

    # Calculate and print accuracies
    print(f"\nEnsemble Model Results ({weight_config} weights):")
    print("-" * 60)
    for name, metrics in results.items():
        accuracy = (metrics['correct'] / metrics['total']) * 100
        print(f"{name:20} Accuracy: {accuracy:.2f}%")

    return results


def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Get test dataloader
    test_loader = get_cifar100_dataloader(batch_size=32, train=False)

    # Test configurations
    configs = ['best', 'epoch5']
    all_results = {}

    for config in configs:
        print(f"\nTesting {config} weights configuration...")
        try:
            # Initialize ensemble model with current configuration
            ensemble_model = EnsembleModel(weight_config=config, device=device)
            # Evaluate ensemble model
            results = evaluate_ensemble(ensemble_model, test_loader, config, device)
            all_results[config] = results
        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue

    # Print comparison if both configurations were successful
    if len(all_results) == 2:
        print("\nAccuracy Comparison (Best vs Epoch 5):")
        print("-" * 60)
        for strategy in ['Max Probability', 'Probability Averaging', 'Majority Voting']:
            best_acc = (all_results['best'][strategy]['correct'] /
                        all_results['best'][strategy]['total']) * 100
            epoch5_acc = (all_results['epoch5'][strategy]['correct'] /
                          all_results['epoch5'][strategy]['total']) * 100
            diff = best_acc - epoch5_acc
            print(f"{strategy:20} Best: {best_acc:.2f}%  Epoch5: {epoch5_acc:.2f}%  Diff: {diff:+.2f}%")


if __name__ == "__main__":
    main()