from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters, parameters_to_ndarrays

from flower_scratch.task import Net, set_weights

import torch

import json
import wandb

class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.results_to_save = {}

        # name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # wandb.init(project="flower-app", name=f"custom-strategy-{name}")

    def aggregate_fit(self,
                      server_round: int,
                      results: list[tuple[ClientProxy, FitRes]],
                      failures: list[tuple[ClientProxy, FitRes] | BaseException]
                      ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        
        # Here, we are just using the aggregate_fit method from the parent class
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        # NOTE: Our ojective is to save the global model at the end of each round
        ndarrays = parameters_to_ndarrays(parameters_aggregated)

        model = Net()
        set_weights(model, ndarrays)

        torch.save(model.state_dict(), f"global_model_{server_round}.pth")

        # In order to maintain the same return signature as the parent class, we return the aggregated parameters and metrics,
        # as used above in the super().aggregate_fit method
        return parameters_aggregated, metrics_aggregated
    
    def evaluate(self,
                 server_round: int,
                 parameters: Parameters
                 ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        loss, metrics = super().evaluate(server_round, parameters)

        my_results = {"loss": loss, **metrics}

        self.results_to_save[server_round] = my_results

        with open(f"results.json", "w") as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        # Log to W&B
        # wandb.log(my_results, step=server_round)

        return loss, metrics
