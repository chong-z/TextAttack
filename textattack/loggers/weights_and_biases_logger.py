from textattack.shared.utils import html_table_from_rows

from .logger import Logger


class WeightsAndBiasesLogger(Logger):
    """Logs attack results to Weights & Biases."""

    def __init__(self, filename="", stdout=False):
        self.init_wandb()
        self._result_table_rows = []

    def __setstate__(self, state):
        self.__dict__ = state
        self.init_wandb()

    def init_wandb(self):
        import wandb
        self.wandb = wandb
        self.wandb.init(project="textattack", resume=True)

    def log_summary_rows(self, rows, title, window_id):
        num_columns = len(rows[0])
        table = self.wandb.Table(columns=[title] + [f"C{i}" for i in range(1, num_columns)])
        for row in rows:
            table.add_data(*row)
        self.wandb.log({window_id: table})
        if num_columns == 2:
            for row in rows:
                metric_name, metric_score = row
                self.wandb.run.summary[metric_name] = metric_score

    def _log_result_table(self):
        """Weights & Biases doesn't have a feature to automatically aggregate
        results across timesteps and display the full table.

        Therefore, we have to do it manually.
        """
        result_table = html_table_from_rows(
            self._result_table_rows, header=["", "Original Input", "Perturbed Input"]
        )
        self.wandb.log({"results": self.wandb.Html(result_table)})

    def log_attack_result(self, result):
        original_text_colored, perturbed_text_colored = result.diff_color(
            color_method="html"
        )
        result_num = len(self._result_table_rows)
        self._result_table_rows.append(
            [
                f"<b>Result {result_num}</b>",
                original_text_colored,
                perturbed_text_colored,
            ]
        )
        result_diff_table = html_table_from_rows(
            [[original_text_colored, perturbed_text_colored]]
        )
        result_diff_table = self.wandb.Html(result_diff_table)
        self.wandb.log(
            {
                "result": result_diff_table,
                "original_output": result.original_result.output,
                "perturbed_output": result.perturbed_result.output,
            }
        )
        self._log_result_table()

    def log_sep(self):
        self.fout.write("-" * 90 + "\n")
