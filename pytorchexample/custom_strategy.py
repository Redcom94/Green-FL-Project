import io
import time
from logging import INFO
from pathlib import Path
from typing import Callable, Iterable, Optional, Any

import torch
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log, logger
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg, Result, FedYogi, FedAdam, FedAdagrad, FedProx
from flwr.serverapp.strategy.strategy_utils import log_strategy_start_info

PROJECT_NAME = "FLOWER-advanced-pytorch"

class CustomStrategyMixin:
    """
    W&B logging, and model checkpointing for Flower strategies.
    """
    save_path: Path
    best_acc_so_far: float
    def set_save_path(self, path: Path):
        """Set the path where wandb logs and model checkpoints will be saved."""
        self.save_path = path

    def _update_best_acc(self, current_round: int, accuracy: float, arrays: ArrayRecord) -> None:
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            logger.log(INFO, "💡 New best global model found: %f", accuracy)
            
            # On utilise un nom de fichier fixe pour écraser l'ancien
            file_name = "best_model.pth" 
            torch.save(arrays.to_torch_state_dict(), self.save_path / file_name)
            
            # On peut aussi sauvegarder un petit fichier texte pour garder trace du round
            with open(self.save_path / "best_model_info.txt", "w") as f:
                f.write(f"Round: {current_round}, Accuracy: {accuracy}")

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[
            Callable[[int, ArrayRecord], Optional[MetricRecord]]
        ] = None,
    ) -> Result:
        """Execute the federated learning strategy logging results to W&B and saving them to disk."""

        # Init W&B
        name = f"{str(self.save_path.parent.name)}/{str(self.save_path.name)}-ServerApp"
        wandb.init(project=PROJECT_NAME, name=name, anonymous="allow")

        # Keep track of best acc
        self.best_acc_so_far = 0.0

        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        
        # Evaluate starting global parameters
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # --- TRAINING (CLIENTAPP-SIDE) ---
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round, arrays, train_config, grid,
                ),
                timeout=timeout,
            )

            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round, train_replies,
            )

            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics
                wandb.log(dict(agg_train_metrics), step=current_round)

            # --- EVALUATION (CLIENTAPP-SIDE) ---
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round, arrays, evaluate_config, grid,
                ),
                timeout=timeout,
            )

            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round, evaluate_replies,
            )

            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics
                wandb.log(dict(agg_evaluate_metrics), step=current_round)

            # --- EVALUATION (SERVERAPP-SIDE) ---
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res
                    self._update_best_acc(current_round, res["accuracy"], arrays)
                    wandb.log(dict(res), step=current_round)

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result
    
    """ les strategy fedavg,fedproxy permettent de calculer les moyennes des poids et tous mais ils ignorent complement les nouvelles metriques client_cpu et client_ram"""
    def aggregate_train(self, current_round: int, results: Iterable[Message]):
        """Récupère les poids ET les métriques psutil des clients."""
        
        # 1. Appeler la méthode originale pour agréger les poids du modèle
        # (On utilise super() pour que FedAvg/FedProx fasse son travail habituel)
        agg_arrays, agg_metrics = super().aggregate_train(current_round, results)

        #super() signifie on cherche d'abrd l'agregation des poids a l'aide de de fedavg 

        # 2. Extraire les données psutil envoyées par les clients
        cpus = []
        rams = []
        
        for msg in results:
            # On cherche les clés que tu as définies dans client_app.py
            metrics = msg.content.get("metrics", {})
            if "client_cpu" in metrics:
                cpus.append(metrics["client_cpu"])
            if "client_ram" in metrics:
                rams.append(metrics["client_ram"])

        # 3. Calculer la moyenne et logger dans le terminal + W&B
        if cpus and rams:
            avg_cpu = sum(cpus) / len(cpus)
            avg_ram = sum(rams) / len(rams)
            
            log(INFO, "\t└──> [Client Monitoring] Avg CPU: %.2f%% | Avg RAM: %.2f%%", avg_cpu, avg_ram)
            
            # Ajouter ces moyennes aux métriques pour qu'elles apparaissent dans W&B
            if agg_metrics is None:
                agg_metrics = MetricRecord({"avg_cpu": avg_cpu, "avg_ram": avg_ram})
            else:
                agg_metrics["avg_cpu"] = avg_cpu
                agg_metrics["avg_ram"] = avg_ram

        return agg_arrays, agg_metrics
# --- Strategy Implementations ---
class CustomFedAvg(CustomStrategyMixin, FedAvg):
    pass

class CustomFedAdagrad(CustomStrategyMixin, FedAdagrad):
    pass


class CustomFedAdam(CustomStrategyMixin, FedAdam):
    pass

class CustomFedYogi(CustomStrategyMixin, FedYogi):
    pass

class CustomFedProx(CustomStrategyMixin, FedProx):
    pass