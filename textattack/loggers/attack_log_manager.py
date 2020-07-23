import numpy as np

from textattack.attack_results import FailedAttackResult, SkippedAttackResult, SuccessfulAttackResult

from . import CSVLogger, FileLogger, VisdomLogger, WeightsAndBiasesLogger

# import torch


class AttackLogManager:
    """Logs the results of an attack to all attached loggers."""

    def __init__(self):
        self.loggers = []
        self.results = []

    def enable_stdout(self):
        self.loggers.append(FileLogger(stdout=True))

    def enable_visdom(self):
        self.loggers.append(VisdomLogger())

    def enable_wandb(self):
        self.loggers.append(WeightsAndBiasesLogger())

    def add_output_file(self, filename):
        self.loggers.append(FileLogger(filename=filename))

    def add_output_csv(self, filename, color_method):
        self.loggers.append(CSVLogger(filename=filename, color_method=color_method))

    def log_result(self, result):
        """Logs an ``AttackResult`` on each of `self.loggers`."""
        self.results.append(result)
        for logger in self.loggers:
            logger.log_attack_result(result)

    def log_results(self, results):
        """Logs an iterable of ``AttackResult`` objects on each of
        `self.loggers`."""
        for result in results:
            self.log_result(result)
        self.log_summary()

    def log_summary_rows(self, rows, title, window_id):
        for logger in self.loggers:
            logger.log_summary_rows(rows, title, window_id)

    def log_sep(self):
        for logger in self.loggers:
            logger.log_sep()

    def flush(self):
        for logger in self.loggers:
            logger.flush()

    def log_attack_details(self, attack, model_name):
        # @TODO log a more complete set of attack details
        attack_detail_rows = [
            ["Attack algorithm:", str(attack)],
            ["Model name:", model_name],
        ]
        self.log_summary_rows(attack_detail_rows, "Attack Details", "attack_details")

    def log_extra_stats(self):
        from collections import defaultdict
        extra_stats_dict = {
            "Original Failed": defaultdict(list),
            "Original Successful": defaultdict(list),
            "Perturbed Failed": defaultdict(list),
            "Perturbed Successful": defaultdict(list),
        }

        for i, result in enumerate(self.results):
            if isinstance(result, FailedAttackResult):
                result_type = "Failed"
            elif isinstance(result, SuccessfulAttackResult):
                result_type = "Successful"
            else:
                continue
            if hasattr(result.original_result, "fill_extra_stats"):
                result.original_result.fill_extra_stats(extra_stats_dict[f"Original {result_type}"])
            if hasattr(result.perturbed_result, "fill_extra_stats"):
                result.perturbed_result.fill_extra_stats(extra_stats_dict[f"Perturbed {result_type}"])

        extra_stats_rows = []
        for result_type, result_dict in extra_stats_dict.items():
            for stats_name, stats_list in result_dict.items():
                avg_value = str(round(np.mean(stats_list), 6))
                extra_stats_rows.append([f"Average {stats_name} ({result_type})", avg_value])
        if len(extra_stats_rows) > 0:
            self.log_summary_rows(extra_stats_rows, f"Extra Stats", "extra_stats")

    def log_summary(self):
        total_attacks = len(self.results)
        if total_attacks == 0:
            return
        # Count things about attacks.
        all_num_words = np.zeros(len(self.results))
        sum_failed_score = 0.0
        sum_successful_score = 0.0
        perturbed_word_counts = np.zeros(len(self.results))
        perturbed_word_percentages = np.zeros(len(self.results))
        num_words_changed_until_success = np.zeros(
            2 ** 16
        )  # @ TODO: be smarter about this
        failed_attacks = 0
        skipped_attacks = 0
        successful_attacks = 0
        max_words_changed = 0
        for i, result in enumerate(self.results):
            all_num_words[i] = len(result.original_result.attacked_text.words)
            if isinstance(result, FailedAttackResult):
                failed_attacks += 1
                sum_failed_score += result.perturbed_result.score
                continue
            elif isinstance(result, SkippedAttackResult):
                skipped_attacks += 1
                continue
            else:
                successful_attacks += 1
                sum_successful_score += result.perturbed_result.score
            num_words_changed = len(
                result.original_result.attacked_text.all_words_diff(
                    result.perturbed_result.attacked_text
                )
            )
            num_words_changed_until_success[num_words_changed - 1] += 1
            max_words_changed = max(
                max_words_changed or num_words_changed, num_words_changed
            )
            if len(result.original_result.attacked_text.words) > 0:
                perturbed_word_percentage = (
                    num_words_changed
                    * 100.0
                    / len(result.original_result.attacked_text.words)
                )
            else:
                perturbed_word_percentage = 0
            perturbed_word_counts[i] = num_words_changed
            perturbed_word_percentages[i] = perturbed_word_percentage

        # Original classifier success rate on these samples.
        original_accuracy = (total_attacks - skipped_attacks) * 100.0 / (total_attacks)
        original_accuracy = str(round(original_accuracy, 2)) + "%"

        # New classifier success rate on these samples.
        accuracy_under_attack = (failed_attacks) * 100.0 / (total_attacks)
        accuracy_under_attack = str(round(accuracy_under_attack, 2)) + "%"

        # Attack success rate.
        if successful_attacks + failed_attacks == 0:
            attack_success_rate = 0
        else:
            attack_success_rate = (
                successful_attacks * 100.0 / (successful_attacks + failed_attacks)
            )
        attack_success_rate = str(round(attack_success_rate, 2)) + "%"

        average_failed_score = 0 if failed_attacks == 0 else sum_failed_score / failed_attacks
        average_failed_score = str(round(average_failed_score, 4))

        average_successful_score = 0 if successful_attacks == 0 else sum_successful_score / successful_attacks
        average_successful_score = str(round(average_successful_score, 4))

        perturbed_word_percentages = perturbed_word_percentages[
            perturbed_word_percentages > 0
        ]
        average_count_words_perturbed = str(round(perturbed_word_counts.mean(), 2))
        average_perc_words_perturbed = perturbed_word_percentages.mean() if len(perturbed_word_percentages) > 0 else 0
        average_perc_words_perturbed = str(round(average_perc_words_perturbed, 2)) + "%"

        average_num_words = all_num_words.mean()
        average_num_words = str(round(average_num_words, 2))

        summary_table_rows = [
            ["Number of successful attacks:", str(successful_attacks)],
            ["Number of failed attacks:", str(failed_attacks)],
            ["Number of skipped attacks:", str(skipped_attacks)],
            ["Original accuracy:", original_accuracy],
            ["Accuracy under attack:", accuracy_under_attack],
            ["Attack success rate:", attack_success_rate],
            ["Average successful score:", average_successful_score],
            ["Average failed score:", average_failed_score],
            ["Average perturbed word #:", average_count_words_perturbed],
            ["Average perturbed word %:", average_perc_words_perturbed],
            ["Average num. words per input:", average_num_words],
        ]

        num_queries = np.array(
            [
                r.num_queries
                for r in self.results
                if not isinstance(r, SkippedAttackResult)
            ]
        )
        avg_num_queries = num_queries.mean()
        avg_num_queries = str(round(avg_num_queries, 2))
        summary_table_rows.append(["Avg num queries:", avg_num_queries])
        self.log_summary_rows(
            summary_table_rows, "Attack Results", "attack_results_summary"
        )
        # Show histogram of words changed.
        numbins = max(max_words_changed, 10)
        for logger in self.loggers:
            logger.log_hist(
                num_words_changed_until_success[:numbins],
                numbins=numbins,
                title="Num Words Perturbed",
                window_id="num_words_perturbed",
            )
