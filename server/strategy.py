# server/strategy.py

import torch
import flwr as fl
from flwr.common import parameters_to_ndarrays
from config import CLIENTS_PER_ROUND, NUM_CLIENTS


class BaseFLStrategy(fl.server.strategy.FedAvg):

    def __init__(self, model_keys, strategy_name, on_round_end=None):
        super().__init__(
            fraction_fit=CLIENTS_PER_ROUND / NUM_CLIENTS,
            fraction_evaluate=CLIENTS_PER_ROUND / NUM_CLIENTS,
            min_fit_clients=CLIENTS_PER_ROUND,
            min_evaluate_clients=CLIENTS_PER_ROUND,
            min_available_clients=NUM_CLIENTS,
        )
        self.model_keys    = model_keys
        self.strategy_name = strategy_name
        self.on_round_end  = on_round_end

    def aggregate_fit(self, server_round, results, failures):
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if aggregated_params is None:
            return None, {}

        # Return aggregated params without loading into model
        # (model is not stored to avoid Ray serialization issues)
        return aggregated_params, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        if results:
            total_examples = sum(r.num_examples for _, r in results)

            avg_accuracy = sum(
                r.metrics["accuracy"] * r.num_examples
                for _, r in results
            ) / total_examples

            total_bytes = sum(
                r.metrics.get("bytes_sent", 0)
                for _, r in results
            )

            print(
                f"[Round {server_round}] "
                f"strategy={self.strategy_name} | "
                f"accuracy={avg_accuracy:.4f} | "
                f"bytes={total_bytes:,}"
            )

            if self.on_round_end:
                self.on_round_end(server_round, avg_accuracy, total_bytes)

        return aggregated_loss, metrics


class FedAvgStrategy(BaseFLStrategy):
    def __init__(self, model_keys, on_round_end=None):
        super().__init__(model_keys, strategy_name="fedavg", on_round_end=on_round_end)


class FedProxStrategy(BaseFLStrategy):
    def __init__(self, model_keys, on_round_end=None):
        super().__init__(model_keys, strategy_name="fedprox", on_round_end=on_round_end)