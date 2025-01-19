from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server.strategy import FedAvg

from torch.utils.data import DataLoader

from flower_scratch.task import Net, get_weights, set_weights, get_transform, test
from flower_scratch.custom_strategy import CustomFedAvg

from datasets import load_dataset

from typing import List, Tuple
import json


def handle_evaluate_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """ 
    A function that aggregate metrics from multiple clients.
    NOTE that this is not the aggregated global model accuracy.
    This accuracy is calculated from the evaluate function on the individual clients' validation set.
    """

    accuracies = [num_samples * m["accuracy"] for num_samples, m in metrics]
    total_samples = sum(num_samples for num_samples, _ in metrics)

    return {
        "accuracy": sum(accuracies) / total_samples,
    }

def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """ Handle arbitrary metrics from clients' training. """

    map_values = []
    for _, m in metrics:
        my_metric_str = m["my_custom_metric"]

        # Convert string to dictionary
        my_metric = json.loads(my_metric_str)
        map_values.append(my_metric["mAP"] * 100)

    return {
        "mAP": sum(map_values) / len(metrics),
    }

def on_fit_config(server_round: int) -> Metrics:
    """
    Config settings from server to client.
    Adjust learning rate based on server round.
    """
    lr = 0.01
    if server_round > 2:
        lr = 0.005

    return {"lr": lr}


def on_evaluate_config(server_round: int) -> Metrics:
    """
    Config settings for evaluate method on client side from server to client.
    """

    return {"just_to_test": server_round}

def get_evaluate_fn(testloader, device):
    """Return a callback that evaluates the global model"""
    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate the global model on the global test set."""
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device)

        return loss, {"cen_accuracy": accuracy}

    return evaluate

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Load global test set
    testset = load_dataset("uoft-cs/cifar10")["test"]

    testloader = DataLoader(testset.with_transform(
        get_transform()), batch_size=32, shuffle=False)
    
    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=handle_evaluate_metrics,
        fit_metrics_aggregation_fn=handle_fit_metrics,
        on_fit_config_fn=on_fit_config,
        on_evaluate_config_fn=on_evaluate_config,
        evaluate_fn=get_evaluate_fn(testloader, device="cpu"),
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
