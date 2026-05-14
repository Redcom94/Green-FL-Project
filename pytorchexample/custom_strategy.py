import io
import time
from logging import INFO
from pathlib import Path
from typing import Callable, Iterable, Optional, Any
import csv
import torch
import wandb
from codecarbon import EmissionsTracker
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
    def configure_train(self, current_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        """Injection du round dans la config avant l'entraînement."""
        # On ajoute le round à la config existante
        config["server_round"] = current_round
        # On appelle la méthode originale de la classe parente (FedAvg, FedProx, etc.)
        return super().configure_train(current_round, arrays, config, grid)
    def configure_evaluate(self, current_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        """Injection du round dans la config avant l'évaluation."""
        config["server_round"] = current_round
        return super().configure_evaluate(current_round, arrays, config, grid)
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
        last_accuracy = 0.0
        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s] - Server Tracking Start", current_round, num_rounds)

            tracker = EmissionsTracker(
                project_name=f"{self.__class__.__name__}_round_{current_round}",
                output_dir=str(self.save_path),
                output_file="emission.csv",
                on_csv_write="update",
                measure_power_secs=1
            )
            tracker.start()

            # Variables réinitialisées à chaque round
            current_acc = None
            emissions_data = None

            try:
                # --- TRAINING (CLIENTAPP-SIDE) ---
                train_replies = grid.send_and_receive(
                    messages=self.configure_train(current_round, arrays, train_config, grid),
                    timeout=timeout,
                )

                agg_arrays, agg_train_metrics = self.aggregate_train(current_round, train_replies)

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
                    if res is not None:
                        current_acc = res["accuracy"]

            finally:
                # --- ARRÊT UNIQUE DU TRACKER ---
                # On arrête toujours ici, qu'il y ait eu une exception ou non.
                tracker.stop()
                emissions_data = tracker.final_emissions  # kg CO2 — valeur réelle après stop()
                log(INFO, "[ROUND %s/%s] - Server Tracking Stop (%.6f kg CO2)", current_round, num_rounds, emissions_data or 0.0)

                # --- CALCUL CARBON PER ACCURACY (après le stop, donc emissions_data est fiable) ---
                if current_acc is not None and emissions_data is not None and emissions_data > 0:
                    acc_gain = current_acc - last_accuracy

                    carbon_per_acc = emissions_data / acc_gain if acc_gain > 0 else 0.0

                    temp_metrics_path = self.save_path / "temp_accuracy_metrics.csv"
                    file_exists = temp_metrics_path.exists()
                    with open(temp_metrics_path, 'a', newline='') as f:
                        writer = csv.writer(f, delimiter=';')
                        if not file_exists:
                            writer.writerow(["round", "carbon_per_accuracy", "accuracy_gain"])
                        writer.writerow([current_round, carbon_per_acc, acc_gain])

                    wandb.log({
                        "round": current_round,
                        "metrics/accuracy_gain": acc_gain,
                        "metrics/carbon_per_accuracy": carbon_per_acc,
                        "env/server_round_co2_kg": emissions_data
                    }, step=current_round)

                    last_accuracy = current_acc

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result
    def aggregate_evaluate(self, current_round: int, results: Iterable[Message]):
        """Agrège les métriques d'évaluation des clients."""
        
        # Filter failures here as well!
        successful_results = [msg for msg in results if msg.has_content()]
        
        if not successful_results:
            return None

        # Aggregate using parent logic
        agg_metrics = super().aggregate_evaluate(current_round, successful_results)

        if agg_metrics is None:
            return None

        if not isinstance(agg_metrics, (dict, MetricRecord)):
            agg_metrics = MetricRecord(agg_metrics)

        # Calculate F1 only for successful results
        f1_scores = []
        example_counts = []

        for msg in successful_results:
            m = msg.content.get("metrics", {})
            if "eval_f1" in m and "num-examples" in m:
                f1_scores.append(float(m["eval_f1"]))
                example_counts.append(int(m["num-examples"]))

        if f1_scores:
            total_examples = sum(example_counts)
            if total_examples > 0:
                weighted_f1 = sum(f * e for f, e in zip(f1_scores, example_counts)) / total_examples
                log(INFO, "\t└──> [Global Evaluation] Aggregated F1-Score: %.4f", weighted_f1)
                agg_metrics["f1_score"] = weighted_f1

        return agg_metrics
    """ les strategy fedavg,fedproxy permettent de calculer les moyennes des poids et tous mais ils ignorent complement les nouvelles metriques client_cpu et client_ram"""
    def aggregate_train(self, current_round: int, results: Iterable[Message]):
        """Récupère les poids ET les métriques psutil des clients."""
        
        # 1. Filter out failures: Keep only messages that have content
        # In the new API, we just check msg.has_content()
        successful_results = [msg for msg in results if msg.has_content()]

        if not successful_results:
            log(INFO, "No successful results to aggregate in round %s", current_round)
            return None, {}

        # 2. Call the parent aggregation (FedAvg/Adam/etc.) with ONLY successful results
        agg_arrays, agg_metrics = super().aggregate_train(current_round, successful_results)

        # 3. Extract psutil data from the filtered list
        cpus = []
        rams = []
        
        for msg in successful_results:
            metrics = msg.content.get("metrics", {})
            if "client_cpu" in metrics:
                cpus.append(metrics["client_cpu"])
            if "client_ram" in metrics:
                rams.append(metrics["client_ram"])

        # 4. Calculate averages
        if cpus and rams:
            avg_cpu = sum(cpus) / len(cpus)
            avg_ram = sum(rams) / len(rams)
            log(INFO, "\t└──> [Client Monitoring] Avg CPU: %.2f%% | Avg RAM: %.2f%%", avg_cpu, avg_ram)
            
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
